import numpy as np
import scipy.io as scio
from code1.models.bnmf_vb import bnmf_vb
from numpy import linalg as LA
import datetime
from numpy import linalg as LA
import math
import numpy, itertools, os
import numpy as np
import random


def SNR(x1, x2):
    from numpy.linalg import norm
    return 20 * np.log10(norm(x1) / norm(x2))


# error = []
starttime = datetime.datetime.now()
total_Error = np.zeros([10, 3])
total_Cost = np.zeros([10, 3])
for x in xrange(1, 2):
    print("{} -th iteration:".format(x))
    alphatau, betatau = 3, 1
    alpha0, beta0 = 6, 2
    maxIter = 300
    minIter = 50

    a0, b0 = 1e-4, 1e-4
    alpha = 1
    tol = 1e-5
    threshold = 1e-3
    
    hyperparams = {'alphatau': alphatau, 'betatau': betatau, 'alpha0': alpha0, 'beta0': beta0, 'alpha': alpha, 'a0': a0,
                   'b0': b0, 'maxIter': maxIter, 'minIter': minIter, 'tol': tol}
    ARD = True
    K = 64
    error_array = []
    cost_array = []
    r_array = []
    remain_k_array = []
    # Total_error = []
    for r in [5]:
        print("r= ", r)
        X_Ori = scio.loadmat('Syntheticdata/X_Ori_{}_R{}.mat'.format(x, r))['X_Ori'].astype(float)
        X_Noi = scio.loadmat('Syntheticdata/X_Noi_Sparse_{}_R{}.mat'.format(x,r))['X_Noi_Sparse'].astype(float)
        I, J = X_Noi.shape
        M = np.ones((I, J))
        # for r in [4]:
        # print("r is: {}".format(r))
        r_array.append(r)
        model = bnmf_vb(X_Noi, M, r, K, ARD, threshold, hyperparams)
        model.initialise()
        result, remain_k = model.run(iterations=maxIter)
        remain_k_array.append(remain_k)

        Error = LA.norm((X_Ori - result), 'fro') / LA.norm(X_Ori, 'fro')
        cost = LA.norm((X_Ori - result), 'fro') ** 2 / float(2);
        # print("Error is: {}".format(Error))
        mixtureSNR = SNR(X_Ori, result - X_Ori)

        error_array.append(Error)
        cost_array.append(cost)
        endtime = datetime.datetime.now()
        print(endtime - starttime).seconds
        print("result is: ")
        print("Error is: {}".format(Error))
        print("Cost is: {}".format(cost))
        print("mixtureSNR is: {}".format(mixtureSNR))

    # error_array


    # print(error_array)
    print(cost_array)
    total_Error[x - 1, :] = np.array(error_array)
    total_Cost[x - 1, :] = np.array(cost_array)

print("---Summary-----")
print(total_Cost)
meanError = np.sum(total_Error, axis=0) / float(10)
meanCost = np.sum(total_Cost, axis=0) / float(10)
print("----Mean Cost----")
print(meanCost)

# vary_r_result[:,1] = np.array(error_array)
scio.savemat('MixtureTotalError.mat', {'totalError': total_Error})
scio.savemat('RealMixturemeanError.mat', {'meanError': meanError})