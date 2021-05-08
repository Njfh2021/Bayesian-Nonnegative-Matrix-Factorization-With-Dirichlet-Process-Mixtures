import numpy as np
import scipy.io as scio
from code1.models.bnmf_vb_ARD import bnmf_vb_ARD
from numpy import linalg as LA

# x=2
# X_Ori = scio.loadmat('CaoyuanData/X_Ori_{}.mat'.format(x))['X_Ori'].astype(float)
# X_Noi = scio.loadmat('CaoyuanData/X_Noi_Mixture_{}.mat'.format(x))['X_Noi_Mixture'].astype(float)
realR = 15
dataset = 1
X_Ori = scio.loadmat('Syntheticdata\X_Ori_{}_R{}.mat'.format(dataset, realR))['X_Ori'].astype(float)
# X_Noi = scio.loadmat('X_Noi_Mixture.mat')['X_Noi_Mixture'].astype(float)
X_Noi = scio.loadmat('Syntheticdata\X_Noi_Mixture_{}_R{}.mat'.format(dataset, realR))['X_Noi_Mixture'].astype(float)
print("Running Real R: {}".format(realR))
I, J = X_Noi.shape
# print("All good till Here")
# test = raw_input()
####Below are initialize the model
# lambdaU = np.ones((I, R))
# lambdaV = np.ones((J, R))
alphatau, betatau = 3, 1
alpha0, beta0 = 6, 2
maxIter = 300
minIter = 50

a0, b0 = 1e-4, 1e-4
alpha = 1
tol = 1e-5
threshold = 1e-3

# R5 = 2*np.ones((I,J))
M = np.ones((I, J))
hyperparams = {'alphatau': alphatau, 'betatau': betatau, 'alpha0': alpha0, 'beta0': beta0, 'alpha': alpha, 'a0': a0,
               'b0': b0, 'maxIter': maxIter, 'minIter': minIter, 'tol': tol}
ARD = True

# R=15
min_R = 35
max_R = 35

# K=64


r_array = []
remain_k_array = []
#K_List = [50, 100, 150]
K_List = [100]
# K_List = [30]
r_total_Error = np.zeros([max_R - min_R + 1, len(K_List)])
count = 0
for r in xrange(min_R, max_R + 1):
    k_error_array = []
    for k in K_List:
        print("r is: {}".format(r))
        r_array.append(r)
        model = bnmf_vb_ARD(X_Noi, M, r, k, ARD, threshold,tol, hyperparams)
        model.initialise()
        result, remain_k = model.run(iterations=maxIter)
        remain_k_array.append(remain_k)

        Error = LA.norm((X_Ori - result), 'fro') / LA.norm(X_Ori, 'fro')
        print("Error is: {}".format(Error))
        k_error_array.append(Error)
    r_total_Error[count, :] = np.array(k_error_array)
    count += 1

scio.savemat('r_total_ErrorR{}.mat'.format(realR), {'r_total_Error': r_total_Error})
# vary_r_result = np.zeros([max_R-min_R+1,2])
# vary_r_result[:,0] = np.array(r_array)
# vary_r_result[:,1] = np.array(error_array)
# scio.savemat('vary_r_result.mat', {'vary_r_result':vary_r_result})


# Test R is 2-dim
# with pytest.raises(AssertionError) as error:
#     bnmf_vb(R1, M, R, ARD, hyperparams)
# assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 1-dimensional."
