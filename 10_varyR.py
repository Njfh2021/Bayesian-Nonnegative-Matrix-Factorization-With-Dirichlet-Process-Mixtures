import numpy as np
import scipy.io as scio
from code.models.bnmf_vb import bnmf_vb
from numpy import linalg as LA

# error = []
initialRList = range(15, 35)
initialRNum = len(initialRList)
total_Error = np.zeros([10, initialRNum])
total_Cost = np.zeros([10, initialRNum])

R=15
for x in xrange(1, 11):
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
    for r in [R]:
        X_Ori = scio.loadmat('Syntheticdata/X_Ori_{}_R{}.mat'.format(x, r))['X_Ori'].astype(float)
        X_Noi = scio.loadmat('Syntheticdata/X_Noi_Mixture_{}_R{}.mat'.format(x,r))['X_Noi_Mixture'].astype(float)
        I, J = X_Noi.shape
        M = np.ones((I, J))
        # for r in [4]:
        # print("r is: {}".format(r))
        r_array.append(r)
        for initialR in initialRList:

            model = bnmf_vb(X_Noi, M, initialR, K, ARD, threshold, hyperparams)
            model.initialise()
            result, remain_k = model.run(iterations=maxIter)
            
            Error = LA.norm((X_Ori - result), 'fro') / LA.norm(X_Ori, 'fro')
            #cost = LA.norm((X_Ori - result), 'fro') ** 2 / float(2);
            # print("Error is: {}".format(Error))
            error_array.append(Error)
            #cost_array.append(cost)
    # error_array

    # print(error_array)
    print(cost_array)
    total_Error[x - 1, :] = np.array(error_array)
    total_Cost[x - 1, :] = np.array(cost_array)

print("---Summary-----")
print(total_Cost)
meanError = np.sum(total_Error, axis=0) / float(10)
#meanCost = np.sum(total_Cost, axis=0) / float(10)
#print("----Mean Cost----")
#print(meanCost)

# vary_r_result[:,1] = np.array(error_array)
scio.savemat('MixtureTotalError_R{}.mat'.format(R), {'totalError': total_Error})
#scio.savemat('RealMixturemeanError.mat', {'meanError': meanError})
