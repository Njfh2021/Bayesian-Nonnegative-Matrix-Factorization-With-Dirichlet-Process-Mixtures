import numpy as np
import scipy.io as scio
from code.models.bnmf_vb import bnmf_vb
from numpy import linalg as LA

X_Ori = scio.loadmat('X_Ori.mat')['X_Ori'].astype(float)
X_Noi = scio.loadmat('X_Noi_Gaussian.mat')['X_Noi_Gaussian'].astype(float)
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

# R5 = 2*np.ones((I,J))
M = np.ones((I, J))
hyperparams = {'alphatau': alphatau, 'betatau': betatau, 'alpha0': alpha0, 'beta0': beta0, 'alpha': alpha, 'a0': a0,
               'b0': b0, 'maxIter': maxIter, 'minIter': minIter, 'tol': tol}
ARD = True

R = 4
# K=64
Min_K = 10
Max_K = 200
# Tried 1e-2 however, it will make all the to zero
threshold = 2e-3

max_Iter = 1
vary_k_5_result = np.zeros([Max_K - Min_K + 1, max_Iter])
for x in xrange(0, max_Iter):
    # pass
    error_array = []
    k_array = []
    remain_k_array = []
    for k in xrange(Min_K, Max_K + 1):
        k_array.append(k)
        model = bnmf_vb(X_Noi, M, R, k, ARD, threshold, hyperparams)
        model.initialise()
        result, remain_k = model.run(iterations=maxIter)

        # MSE = model.compute_MSE(M,X_Ori,result)
        # print("MSE is: {}".format(MSE))
        Error = LA.norm((X_Ori - result), 'fro') / LA.norm(X_Ori, 'fro')
        print("Error : {}".format(Error))
        print("remain_k: {}".format(remain_k))
        error_array.append(Error)
        remain_k_array.append(remain_k)

    vary_k_5_result[:, x] = np.array(error_array)
# vary_k_5_result[:,1] = np.array(error_array)
# vary_k_5_result[:,2] = np.array(remain_k_array)

scio.savemat('vary_k_5_result.mat', {'vary_k_5_result': vary_k_5_result})

# Test R is 2-dim
# with pytest.raises(AssertionError) as error:
#     bnmf_vb(R1, M, R, ARD, hyperparams)
# assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 1-dimensional."
