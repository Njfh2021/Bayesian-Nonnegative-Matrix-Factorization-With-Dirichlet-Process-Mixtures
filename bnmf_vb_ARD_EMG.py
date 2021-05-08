"""
Variational Bayesian inference for non-negative matrix factorisation, with ARD.

We expect the following arguments:
- D, the matrix.
- M, the mask matrix indicating observed values (1) and unobserved ones (0).
- R, the number of latent factors.
- ARD, a boolean indicating whether we use ARD in this model or not.
- hyperparameters = { 'alphatau', 'betatau', 'alpha0', 'beta0', 'lambdaU', 'lambdaV' },
    alphatau, betatau - non-negative reals defining prior over noise parameter tau.
    alpha0, beta0     - if using the ARD, non-negative reals defining prior over ARD lambda.
    lambdaU, lambdaV  - if not using the ARD, nonnegative reals defining prior over U and V
    
The random variables are initialised as follows:
    (lambdar) alphar_s, betar_s - set to alpha0, beta0
    (U,V) muU, muV - expectation ('exp') or random ('random')
    (U,V) tauU, tauV - set to 1
    (tau) alpha_s, beta_s - using updates
We initialise the values of U and V according to the given argument 'init_UV'. 

Usage of class:
    BNMF = bnmf_vb(D,M,R,ARD,hyperparameters)
    BNMF.initisalise(init_UV)      
    BNMF.run(iterations)
Or:
    BNMF = bnmf_vb(D,M,R,ARD,hyperparameters)
    BNMF.train(init_UV,iterations)

We can test the performance of our model on a test dataset, specifying our test set with a mask M. 
    performance = BNMF.predict(M_pred)
This gives a dictionary of performances,
    performance = { 'MSE', 'R^2', 'Rp' }
    
The performances of all iterations are stored in BNMF.all_performances, which 
is a dictionary from 'MSE', 'R^2', or 'Rp' to a list of performances.
    
Finally, we can return the goodness of fit of the data using the quality(metric) function:
- metric = 'loglikelihood' -> return p(D|theta)
         = 'BIC'        -> return Bayesian Information Criterion
         = 'AIC'        -> return Afaike Information Criterion
         = 'MSE'        -> return Mean Square Error
         = 'ELBO'       -> return Evidence Lower Bound
"""

import itertools
import math
import numpy
import numpy as np
import scipy
import time
from distributions.exponential import exponential_draw
from distributions.gamma import gamma_expectation, gamma_expectation_log
from distributions.truncated_normal_vector import TN_vector_expectation, TN_vector_variance
from scipy.special import psi

import scipy.io as scio

# ALL_METRICS = ['MSE','R^2','Rp']
ALL_METRICS = ['MSE', 'R^2']
ALL_QUALITY = ['loglikelihood', 'BIC', 'AIC', 'MSE', 'ELBO']
OPTIONS_INIT_UV = ['random', 'exp']


class bnmf_vb_ARD_EMG:
    def __init__(self, D, M, R, K, ARD, thre_K, thre_Lambda, hyperparameters):
        ''' Set up the class and do some checks on the values passed. '''
        self.D = numpy.array(D, dtype=float)
        self.M = numpy.array(M, dtype=float)
        self.R = R
        self.K = K
        self.ARD = ARD
        self.thre_K = thre_K
        self.thre_Lambda = thre_Lambda

        assert len(self.D.shape) == 2, "Input matrix R is not a two-dimensional array, " \
                                       "but instead %s-dimensional." % len(self.D.shape)
        assert self.D.shape == self.M.shape, "Input matrix R is not of the same size as " \
                                             "the indicator matrix M: %s and %s respectively." % (
                                             self.D.shape, self.M.shape)

        (self.I, self.J) = self.D.shape
        self.size_Omega = self.M.sum()
        self.check_empty_rows_columns()

        ########From Here on From AMF Initialize
        self.alphatau, self.betatau = float(hyperparameters['alphatau']), float(hyperparameters['betatau'])
        self.a0, self.b0 = float(hyperparameters['a0']), float(hyperparameters['b0'])
        self.minIter, self.maxIter = int(hyperparameters['minIter']), float(hyperparameters['maxIter'])
        self.alpha, self.tol = float(hyperparameters['alpha']), float(hyperparameters['tol'])

        #####End of AMF initialize

        if self.ARD:
            self.alpha0, self.beta0 = float(hyperparameters['alpha0']), float(hyperparameters['beta0'])
        else:
            self.lambdaU, self.lambdaV = numpy.array(hyperparameters['lambdaU']), numpy.array(
                hyperparameters['lambdaV'])
            # Make lambdaU/V into a numpy array if they are an integer
            if self.lambdaU.shape == ():
                self.lambdaU = self.lambdaU * numpy.ones((self.I, self.R))
            if self.lambdaV.shape == ():
                self.lambdaV = self.lambdaV * numpy.ones((self.J, self.R))

            assert self.lambdaU.shape == (
            self.I, self.R), "Prior matrix lambdaU has the wrong shape: %s instead of (%s, %s)." % (
            self.lambdaU.shape, self.I, self.R)
            assert self.lambdaV.shape == (
            self.J, self.R), "Prior matrix lambdaV has the wrong shape: %s instead of (%s, %s)." % (
            self.lambdaV.shape, self.J, self.R)

    def check_empty_rows_columns(self):
        ''' Raise an exception if an entire row or column is empty. '''
        sums_columns = self.M.sum(axis=0)
        sums_rows = self.M.sum(axis=1)

        # Assert none of the rows or columns are entirely unknown values
        for i, c in enumerate(sums_rows):
            assert c != 0, "Fully unobserved row in R, row %s." % i
        for j, c in enumerate(sums_columns):
            assert c != 0, "Fully unobserved column in R, column %s." % j

    def train(self, init_UV, iterations):
        ''' Initialise and run the algorithm. '''
        self.initialise(init_UV=init_UV)
        self.run(iterations)

    def initialise(self, init_UV='exp'):
        ''' Initialise U, V, tau, and lambda (if ARD). '''
        assert init_UV in OPTIONS_INIT_UV, "Unknown initialisation option: %s. Should be in %s." % (
        init_UV, OPTIONS_INIT_UV)
        # By Caoyuan parameters for q_\beta(\gamma_1, \gamma_2)
        self.Gamma = np.ones([self.K, 2])
        self.Gamma[:, 1] = self.alpha * np.ones(self.K)
        # By Caoyuan parameters for q_sigma_k (\tau_1, \tau_2)
        self.Tao = self.a0 * np.ones([self.K, 2])
        self.Tao[:, 1] = self.b0 * np.ones(self.K)

        self.exp_Tau = self.Tao[:, 0] / self.Tao[:, 1]

        # By Caoyuan parameters for q_z_mn(\phi)
        self.Phi = 1 / float(self.K) * np.ones([self.K, self.I, self.J])
        # print(self.Phi)
        # test = raw_input()

        # Initialise lambdar, and compute expectation
        if self.ARD:
            self.alphar_s, self.betar_s = numpy.zeros(self.R), numpy.zeros(self.R)
            self.exp_lambdar, self.exp_loglambdar = numpy.zeros(self.R), numpy.zeros(self.R)
            for r in range(self.R):
                self.alphar_s[r] = self.alpha0
                self.betar_s[r] = self.beta0
                self.update_exp_lambdar(r)

        # Initialise parameters U, V
        self.mu_U, self.tau_U = numpy.zeros((self.I, self.R)), numpy.zeros((self.I, self.R))
        self.mu_V, self.tau_V = numpy.zeros((self.J, self.R)), numpy.zeros((self.J, self.R))

        for i, r in itertools.product(range(self.I), range(self.R)):
            self.tau_U[i, r] = 1.
            hyperparam = self.exp_lambdar[r] if self.ARD else self.lambdaU[i, r]
            self.mu_U[i, r] = exponential_draw(hyperparam) if init_UV == 'random' else 1.0 / hyperparam
        for j, r in itertools.product(range(self.J), range(self.R)):
            self.tau_V[j, r] = 1.
            hyperparam = self.exp_lambdar[r] if self.ARD else self.lambdaV[j, r]
            self.mu_V[j, r] = exponential_draw(hyperparam) if init_UV == 'random' else 1.0 / hyperparam

        # Compute expectations and variances U, V
        self.exp_U, self.var_U = numpy.zeros((self.I, self.R)), numpy.zeros((self.I, self.R))
        self.exp_V, self.var_V = numpy.zeros((self.J, self.R)), numpy.zeros((self.J, self.R))

        for r in range(self.R):
            self.update_exp_U(r)
        for r in range(self.R):
            self.update_exp_V(r)

        # Initialise tau and compute expectation
        # By Caoyuan here means tau_k
        # self.update_tau()
        # self.update_exp_tau()

    def run(self, iterations):
        ''' Run the Gibbs sampler. '''
        # self.all_exp_tau = []  # to check for convergence
        self.all_times = []  # to plot performance against time

        self.all_performances = {}  # for plotting convergence of metrics
        for metric in ALL_METRICS:
            self.all_performances[metric] = []

        time_start = time.time()
        #f = open('R500Exp','w')
        for it in range(iterations):
            #print("Iteration: {}".format(it))
            # Update lambdar
            if self.ARD:
                for r in range(self.R):
                    self.update_lambdar(r)
                    self.update_exp_lambdar(r)
            #By Caoyuan, start to prune r
            #print("===self.exp_lambdar is: {}".format(self.exp_lambdar))
            flagR = self.exp_lambdar < self.thre_Lambda
            #f.write(str(self.exp_lambdar)+'\n')
            
            l = sum(flagR)
            if l==0:
                print('Error Here!, so we do not update R')
                print(self.exp_lambdar)
            elif it==iterations-1:
                self.R = len(self.exp_lambdar)
                self.exp_lambdar = self.exp_lambdar[flagR]
                self.exp_loglambdar = self.exp_loglambdar[flagR]
                self.alphar_s = self.alphar_s[flagR]
                self.betar_s = self.betar_s[flagR]
                self.R = len(self.exp_lambdar)
                self.mu_U = self.mu_U[:, flagR]
                self.tau_U = self.tau_U[:, flagR]

                self.mu_V = self.mu_V[:, flagR]
                self.tau_V = self.tau_V[:, flagR]

            #By Caoyuan, end of Pruning r


            ####By Caoyuan, start of DemoDP
            # err = np.zeros([self.I, self.J])
            err = self.each_exp_square_diff()
            # temp0 = np.dot(self.mu_U, np.transpose(self.mu_V))*self.D
            # err = ?
            # Need to implement err later
            phi = np.zeros(self.K)
            for i in range(self.K):
                phi[i] = sum(sum(self.Phi[i, :, :]))
            Sum_to_k = sum(phi)
            Pai = np.zeros(self.K)
            # By Caoyuan, pai_vec_prod is theta_k
            psi_vec_sum = np.zeros(self.K)
            pai_vec_prod = np.ones(self.K)

            for x in range(self.K):
                self.Gamma[x, 0] = 1 + phi[x]  # Eq8 in AMF, Eq 5 in ours
                Sum_to_k = Sum_to_k - phi[x]
                self.Gamma[x, 1] = self.alpha + Sum_to_k  # Eq 8 line 2 in AMF Eq 6 in ours
                if (x == 0):
                    psi_vec_sum[x] = 0
                    pai_vec_prod[x] = 1
                else:
                    if x == 1:
                        psi_vec_sum[x] = psi(self.Gamma[x - 1, 1]) - psi(self.Gamma[x - 1, 0] + self.Gamma[x - 1, 1])
                        pai_vec_prod[x] = self.Gamma[x - 1, 1] / (self.Gamma[x - 1, 0] + self.Gamma[x - 1, 1])
                        pass
                    else:
                        psi_vec_sum[x] = psi_vec_sum[x - 1] + psi(self.Gamma[x - 1, 1]) - psi(
                            self.Gamma[x - 1, 0] + self.Gamma[x - 1, 1])
                        pai_vec_prod[x] = pai_vec_prod[x - 1] * self.Gamma[x - 1, 1] / (
                                    self.Gamma[x - 1, 0] + self.Gamma[x - 1, 1])

            for k in range(self.K):
                Sum_of_psi = psi_vec_sum[k]
                Prod_of_pai = pai_vec_prod[k]
                # Eq 8 line 3 in AMF, Eq 13 line 1
                self.Tao[k, 0] = self.a0 + 0.5 * phi[k]
                self.Tao[k, 1] = self.b0 + 0.5 * sum(sum(self.Phi[k, :, :] * err))

                if k == self.K - 1:
                    self.Gamma[k, 1] = 0  # By C, Eq 6 in AMF, k cannot be K
                # By C Eq10 in AMF and Eq 10 in ours
                self.Phi[k, :, :] = np.exp(
                    (psi(self.Gamma[k, 0]) - psi(self.Gamma[k, 0] + self.Gamma[k, 1]) + Sum_of_psi) * np.ones(
                        [self.I, self.J]) - 0.5 * self.Tao[k, 0] / self.Tao[k, 1] * err - 0.5 * (
                                np.log(self.Tao[k, 1]) - psi(self.Tao[k, 0])) * np.ones([self.I, self.J]))
                Pai[k] = Prod_of_pai * self.Gamma[k, 0] / (self.Gamma[k, 0] + self.Gamma[k, 1])
                pass

            # By Caoyuan Update the exceptation of Tao, to be used later
            self.exp_Tau = self.Tao[:, 0] / self.Tao[:, 1]
            Pai = Pai / sum(Pai)
            # print(Pai)
            # flag = Pai>1e-4
            flag = Pai > self.thre_K
            # thre = self.thre_K
            # while sum(flag)<2:
            #     thre-=  1e-3
            #     flag = Pai>thre

            # print(Pai)
            lenOfFlag=sum(flag)
            if lenOfFlag==0:
                pass
            else:
                Pai = Pai[flag]
                self.K = len(Pai)
                # print("In Iteration {},K is :{}".format(it, self.K))
                # print(")
                self.Gamma = self.Gamma[flag, :]
                self.Tao = self.Tao[flag, :]
                # Add by Caoyuan Later, exp is not updated before
                self.exp_Tau = self.exp_Tau[flag]
                self.Phi = self.Phi[flag, :, :]
            # What is the axis of sum?
            Phi_sum = sum(self.Phi)
            # print("=============")
            # for x in range(self.I):
            #     for y in range(self.J):
            #         if Phi_sum[x][y] == 0:
            #             print("ERROR at {} row and {} column".format(x + 1, y + 1))
            #     pass
            # print(Phi_sum)
            self.Phi = self.Phi / Phi_sum

            #####By Caoyuan, end of DemoDP

            # Update U
            for r in range(self.R):
                self.update_U(r)
                self.update_exp_U(r)

                # Update V
            for r in range(self.R):
                self.update_V(r)
                self.update_exp_V(r)

            # Update tau
            # By Caoyuan, below two line should be deleted
            # self.update_tau()
            # self.update_exp_tau()

            # Store expectations
            # self.all_exp_tau.append(self.exp_tau)

            # Store and print performances
            # perf, elbo = self.predict(self.M), self.elbo()
            #perf = self.predict(self.M)
            #for metric in ALL_METRICS:
            #    self.all_performances[metric].append(perf[metric])

            # print "Iteration %s. ELBO: %s. MSE: %s. R^2: %s. Rp: %s." % (it+1,elbo,perf['MSE'],perf['R^2'],perf['Rp'])
            # print "Iteration %s. ELBO: %s. MSE: %s. R^2: %s. ." % (it+1,elbo,perf['MSE'],perf['R^2'])

            # Store time taken for iteration
            time_iteration = time.time()
            self.all_times.append(time_iteration - time_start)
        #f.close()
        #scio.savemat('Lambda{}.mat'.format(self.R), {'Lambda': self.exp_lambdar})
        #scio.savemat('UR{}.mat'.format(self.R), {'U': self.exp_U})
        #scio.savemat('VR{}.mat'.format(self.R), {'V': self.exp_V})
        return (self.exp_U, self.exp_V, self.K, self.R)

    def elbo(self):
        ''' Compute the ELBO. '''
        total_elbo = 0.

        # Log likelihood   
        exp_logtau = 1
        exp_tau = 1
        total_elbo += self.size_Omega / 2. * (exp_logtau - math.log(2 * math.pi)) \
                      - exp_tau / 2. * self.exp_square_diff()

        # Prior lambdar, if using ARD, and prior U, V
        if self.ARD:
            total_elbo += self.alpha0 * math.log(self.beta0) - scipy.special.gammaln(self.alpha0) \
                          + (self.alpha0 - 1.) * self.exp_loglambdar.sum() - self.beta0 * self.exp_lambdar.sum()

            total_elbo += self.I * numpy.log(self.exp_lambdar).sum() - (self.exp_lambdar * self.exp_U).sum()
            total_elbo += self.J * numpy.log(self.exp_lambdar).sum() - (self.exp_lambdar * self.exp_V).sum()

        else:
            total_elbo += numpy.log(self.lambdaU).sum() - (self.lambdaU * self.exp_U).sum()
            total_elbo += numpy.log(self.lambdaV).sum() - (self.lambdaV * self.exp_V).sum()

        # Prior tau
        exp_tau = 1
        total_elbo += self.alphatau * math.log(self.betatau) - scipy.special.gammaln(self.alphatau) \
                      + (self.alphatau - 1.) * exp_logtau - self.betatau * exp_tau

        # q for lambdar, if using ARD
        if self.ARD:
            total_elbo += - sum([v1 * math.log(v2) for v1, v2 in zip(self.alphar_s, self.betar_s)]) + sum(
                [scipy.special.gammaln(v) for v in self.alphar_s]) \
                          - ((self.alphar_s - 1.) * self.exp_loglambdar).sum() + (self.betar_s * self.exp_lambdar).sum()

        # q for U, V
        total_elbo += - .5 * numpy.log(self.tau_U).sum() + self.I * self.R / 2. * math.log(2 * math.pi) \
                      + numpy.log(0.5 * scipy.special.erfc(-self.mu_U * numpy.sqrt(self.tau_U) / math.sqrt(2))).sum() \
                      + (self.tau_U / 2. * (self.var_U + (self.exp_U - self.mu_U) ** 2)).sum()
        total_elbo += - .5 * numpy.log(self.tau_V).sum() + self.J * self.R / 2. * math.log(2 * math.pi) \
                      + numpy.log(0.5 * scipy.special.erfc(-self.mu_V * numpy.sqrt(self.tau_V) / math.sqrt(2))).sum() \
                      + (self.tau_V / 2. * (self.var_V + (self.exp_V - self.mu_V) ** 2)).sum()

        # q for tau
        # total_elbo += - self.alpha_s * math.log(self.beta_s) + scipy.special.gammaln(self.alpha_s) \
        #              - (self.alpha_s - 1.)*self.exp_logtau + self.beta_s * self.exp_tau

        return total_elbo

    ''' Update the parameters for the distributions. '''
    # def update_tau(self):
    ''' Parameter updates tau. '''

    #    self.alpha_s = self.alphatau + self.size_Omega/2.0
    #    self.beta_s = self.betatau + 0.5*self.exp_square_diff()

    def exp_square_diff(self):
        ''' Compute: sum_Omega E_q(U,V) [ ( Rij - Ui Vj )^2 ]. '''
        return (self.M * ((self.D - numpy.dot(self.exp_U, self.exp_V.T)) ** 2 + \
                          (numpy.dot(self.var_U + self.exp_U ** 2, (self.var_V + self.exp_V ** 2).T) - numpy.dot(
                              self.exp_U ** 2, (self.exp_V ** 2).T)))).sum()

    def each_exp_square_diff(self):
        ''' Compute: sum_Omega E_q(U,V) [ ( Rij - Ui Vj )^2 ]. '''
        return self.M * ((self.D - numpy.dot(self.exp_U, self.exp_V.T)) ** 2 + \
                         (numpy.dot(self.var_U + self.exp_U ** 2, (self.var_V + self.exp_V ** 2).T) - numpy.dot(
                             self.exp_U ** 2, (self.exp_V ** 2).T)))

    def update_lambdar(self, r):
        ''' Parameter updates lambdar. '''
        self.alphar_s[r] = self.alpha0 + self.I + self.J
        self.betar_s[r] = self.beta0 + self.exp_U[:, r].sum() + self.exp_V[:, r].sum()

    def update_U(self, r):
        ''' Parameter updates U. '''
        lamb = self.exp_lambdar[r] if self.ARD else self.lambdaU[:, r]
        # By Caoyuan, Original:
        # self.tau_U[:,r] = self.exp_tau*(self.M*( self.var_V[:,r] + self.exp_V[:,r]**2 )).sum(axis=1) #sum over j, so rows
        # self.mu_U[:,r] = 1./self.tau_U[:,r] * (-lamb + self.exp_tau*(self.M * ( (self.D-numpy.dot(self.exp_U,self.exp_V.T)+numpy.outer(self.exp_U[:,r],self.exp_V[:,r]))*self.exp_V[:,r] )).sum(axis=1))
        # By Caoyuan, New:
        # test = self.M*( self.var_V[:,r] + self.exp_V[:,r]**2 )
        # print("self.exp_V[:,r]**2")
        # print(self.exp_V[:,r]**2)
        # print('========')
        # print(self.var_V[:,r])
        # print("test is: ======")
        # print(test)
        # # test1= raw_input()
        # tau = 1
        self.tau_U[:, r] = (sum(self.Phi * self.exp_Tau[:, np.newaxis, np.newaxis]) * self.M * (
                    self.var_V[:, r] + self.exp_V[:, r] ** 2)).sum(axis=1)  # sum over j, so rows
        self.mu_U[:, r] = 1. / self.tau_U[:, r] * (-lamb + (
                    sum(self.Phi * self.exp_Tau[:, np.newaxis, np.newaxis]) * self.M * ((self.D - numpy.dot(self.exp_U,
                                                                                                            self.exp_V.T) + numpy.outer(
                self.exp_U[:, r], self.exp_V[:, r])) * self.exp_V[:, r])).sum(axis=1))

    def update_V(self, r):
        ''' Parameter updates V. '''
        lamb = self.exp_lambdar[r] if self.ARD else self.lambdaV[:, r]
        # By Caoyuan, Original:
        # self.tau_V[:,r] = self.exp_tau*(self.M.T*( self.var_U[:,r] + self.exp_U[:,r]**2 )).T.sum(axis=0) #sum over i, so columns
        # self.mu_V[:,r] = 1./self.tau_V[:,r] * (-lamb + self.exp_tau*(self.M.T * ( (self.D-numpy.dot(self.exp_U,self.exp_V.T)+numpy.outer(self.exp_U[:,r],self.exp_V[:,r])).T*self.exp_U[:,r] )).T.sum(axis=0))
        # By Caoyuan, New:
        # tau =1
        self.tau_V[:, r] = (sum(self.Phi * self.exp_Tau[:, np.newaxis, np.newaxis]).T * self.M.T * (
                    self.var_U[:, r] + self.exp_U[:, r] ** 2)).T.sum(axis=0)  # sum over i, so columns
        self.mu_V[:, r] = 1. / self.tau_V[:, r] * (-lamb + (
                    sum(self.Phi * self.exp_Tau[:, np.newaxis, np.newaxis]).T * self.M.T * ((self.D - numpy.dot(
                self.exp_U, self.exp_V.T) + numpy.outer(self.exp_U[:, r], self.exp_V[:, r])).T * self.exp_U[:,
                                                                                                 r])).T.sum(axis=0))

    ''' Update the expectations and variances. '''
    # def update_exp_tau(self):
    ''' Update expectation tau. '''

    #    self.exp_tau = gamma_expectation(self.alpha_s,self.beta_s)
    #    self.exp_logtau = gamma_expectation_log(self.alpha_s,self.beta_s)

    def update_exp_lambdar(self, r):
        ''' Update expectation lambdar. '''
        self.exp_lambdar[r] = gamma_expectation(self.alphar_s[r], self.betar_s[r])
        self.exp_loglambdar[r] = gamma_expectation_log(self.alphar_s[r], self.betar_s[r])
        #flagR = Pai > self.thre_K

    def update_exp_U(self, r):
        ''' Update expectation U. '''
        self.exp_U[:, r] = TN_vector_expectation(self.mu_U[:, r], self.tau_U[:, r])
        self.var_U[:, r] = TN_vector_variance(self.mu_U[:, r], self.tau_U[:, r])

    def update_exp_V(self, r):
        ''' Update expectation V. '''
        self.exp_V[:, r] = TN_vector_expectation(self.mu_V[:, r], self.tau_V[:, r])
        self.var_V[:, r] = TN_vector_variance(self.mu_V[:, r], self.tau_V[:, r])

    def predict(self, M_pred):
        ''' Predict missing values in R. '''
        R_pred = numpy.dot(self.exp_U, self.exp_V.T)
        MSE = self.compute_MSE(M_pred, self.D, R_pred)
        R2 = self.compute_R2(M_pred, self.D, R_pred)
        # Rp = self.compute_Rp(M_pred, self.D, R_pred)
        # return {'MSE': MSE, 'R^2': R2, 'Rp': Rp}
        return {'MSE': MSE, 'R^2': R2}

    ''' Functions for computing MSE, R^2 (coefficient of determination), Rp (Pearson correlation) '''

    def compute_MSE(self, M, R, R_pred):
        ''' Return the MSE of predictions in R_pred, expected values in R, for the entries in M. '''
        return (M * (R - R_pred) ** 2).sum() / float(M.sum())

    def compute_R2(self, M, R, R_pred):
        ''' Return the R^2 of predictions in R_pred, expected values in R, for the entries in M. '''
        mean = (M * R).sum() / float(M.sum())
        SS_total = float((M * (R - mean) ** 2).sum())
        SS_res = float((M * (R - R_pred) ** 2).sum())
        return 1. - SS_res / SS_total if SS_total != 0. else numpy.inf

    def compute_Rp(self, M, R, R_pred):
        ''' Return the Rp of predictions in R_pred, expected values in R, for the entries in M. '''
        mean_real = (M * R).sum() / float(M.sum())
        mean_pred = (M * R_pred).sum() / float(M.sum())
        covariance = (M * (R - mean_real) * (R_pred - mean_pred)).sum()
        variance_real = (M * (R - mean_real) ** 2).sum()
        variance_pred = (M * (R_pred - mean_pred) ** 2).sum()
        return covariance / float(math.sqrt(variance_real) * math.sqrt(variance_pred))

    def quality(self, metric):
        ''' Return the model quality, either as log likelihood, BIC, AIC, MSE, or ELBO. '''
        assert metric in ALL_QUALITY, 'Unrecognised metric for model quality: %s.' % metric

        log_likelihood = self.log_likelihood()
        if metric == 'loglikelihood':
            return log_likelihood
        elif metric == 'BIC':
            # -2*loglikelihood + (no. free parameters * log(no data points))
            return - 2 * log_likelihood + self.number_parameters() * math.log(self.size_Omega)
        elif metric == 'AIC':
            # -2*loglikelihood + 2*no. free parameters
            return - 2 * log_likelihood + 2 * self.number_parameters()
        elif metric == 'MSE':
            R_pred = numpy.dot(self.exp_U, self.exp_V.T)
            return self.compute_MSE(self.M, self.D, R_pred)
        elif metric == 'ELBO':
            return self.elbo()

    def log_likelihood(self):
        ''' Return the likelihood of the data given the trained model's parameters. '''
        tau = exp_logtau = 1
        return self.size_Omega / 2. * (exp_logtau - math.log(2 * math.pi)) \
               - tau / 2. * (self.M * (self.D - numpy.dot(self.exp_U, self.exp_V.T)) ** 2).sum()

    def number_parameters(self):
        ''' Return the number of free variables in the model. '''
        return (self.I * self.R + self.J * self.R + 1) + (self.R if self.ARD else 0)
