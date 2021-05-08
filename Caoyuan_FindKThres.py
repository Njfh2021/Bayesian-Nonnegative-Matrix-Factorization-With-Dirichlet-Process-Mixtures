import numpy as np
import scipy.io as scio
from code.models.bnmf_vb import bnmf_vb
from numpy import linalg as LA

X_Ori = scio.loadmat('X_Ori.mat')['X_Ori'].astype(float)
X_Noi = scio.loadmat('X_Noi_Mixture.mat')['X_Noi_Mixture'].astype(float)
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

k_threshold = 1e-5

a0, b0 = 1e-4, 1e-4
alpha = 1
tol = 1e-5

# R5 = 2*np.ones((I,J))
M = np.ones((I, J))
hyperparams = {'alphatau': alphatau, 'betatau': betatau, 'alpha0': alpha0, 'beta0': beta0, 'alpha': alpha, 'a0': a0,
               'b0': b0, 'maxIter': maxIter, 'minIter': minIter, 'tol': tol}
ARD = True

R = 4

min_R = 4
max_R = 23

min_K = 10
max_K = 200

K = 10

error_array = []
k_array = []
thre_array = []

remain_k_array = []
base = 1e-3
min_thre = 1
max_thre = 100
for thre in xrange(min_thre, max_thre + 1):
    thre_k = thre * base
    thre_array.append(thre_k)
    model = bnmf_vb(X_Noi, M, R, K, ARD, thre_k, hyperparams)
    model.initialise()
    result, remain_k = model.run(iterations=maxIter)
    print("threshold is: {}".format(thre))
    print("remain_k is: {}".format(remain_k))
    remain_k_array.append(remain_k)

    # MSE = model.compute_MSE(M,X_Ori,result)
    # print("MSE is: {}".format(MSE))

    Error = LA.norm((X_Ori - result), 'fro') / LA.norm(X_Ori, 'fro')
    print("Error is: {}".format(Error))
    error_array.append(Error)

vary_thre_result = np.zeros([max_thre - min_thre + 1, 2])
vary_thre_result[:, 0] = np.array(thre_array)
vary_thre_result[:, 1] = np.array(remain_k_array)

scio.savemat('vary_thre_result.mat', {'vary_thre_result': vary_thre_result})

# Test R is 2-dim
# with pytest.raises(AssertionError) as error:
#     bnmf_vb(R1, M, R, ARD, hyperparams)
# assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 1-dimensional."
