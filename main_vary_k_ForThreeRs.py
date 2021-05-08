import numpy as np
import scipy.io as scio
from code1.models.bnmf_vb import bnmf_vb
from numpy import linalg as LA
#Run three times for R=5, 10, 15
r_List = [5]
print("Running R: {}".format(r_List[0]))
X_Ori = scio.loadmat('Syntheticdata\X_Ori_1_R{}.mat'.format(r_List[0]))['X_Ori'].astype(float)
X_Noi = scio.loadmat('Syntheticdata\X_Noi_Mixture_1_R{}.mat'.format(r_List[0]))['X_Noi_Mixture'].astype(float)
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

# R=4
# K=64
#min_K = 10
min_K = 200
max_K = 300
# Tried 1e-2 however, it will make all the to zero
threshold = 5e-3


remaink_total = np.zeros([max_K - min_K + 1, len(r_List)], dtype=int)
k_total_Error = np.zeros([max_K - min_K + 1, len(r_List)])
count = 0
for r in r_List:
    error_array = []
    k_array = []
    remain_k_array = []
    for k in xrange(min_K, max_K + 1):
        k_array.append(k)
        model = bnmf_vb(X_Noi, M, r, k, ARD, threshold, hyperparams)
        model.initialise()
        result, remain_k = model.run(iterations=maxIter)

        Error = LA.norm((X_Ori - result), 'fro') / LA.norm(X_Ori, 'fro')
        print("Error : {}".format(Error))
        print("remain_k: {}".format(remain_k))
        error_array.append(Error)
        remain_k_array.append(remain_k)
    k_total_Error[:, count] = np.array(error_array)
    remaink_total[:, count] = np.array(remain_k_array)
    count += 1

# vary_k_result = np.zeros([Max_K-Min_K+1,3])
# vary_k_result[:,0] = np.array(k_array)
# vary_k_result[:,1] = np.array(error_array)
# vary_k_result[:,2] = np.array(remain_k_array)

scio.savemat('k_total_ErrorR{}.mat'.format(r_List[0]), {'k_total_Error': k_total_Error})
scio.savemat('remaink_totalR{}.mat'.format(r_List[0]), {'remaink_total': remaink_total})

# Test R is 2-dim
# with pytest.raises(AssertionError) as error:
#     bnmf_vb(R1, M, R, ARD, hyperparams)
# assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 1-dimensional."
