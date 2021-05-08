# project_location = "/path/to/folder/containing/project/"
# import sys
# sys.path.append(project_location) 
import math
import numpy as np
# from code.models.nmf_np import nmf_np
from code.models.bnmf_vb import bnmf_vb
from numpy import linalg as LA

# def __init__(self,R,M,R,ARD,hyperparameters):
I, J, R = 200, 200, 4
# Below are initialize the dataset
RU = np.random.rand(I, R)
RV = np.random.rand(R, J)
X_Ori = np.dot(RU, RV)

X_Noi = X_Ori
x_Noi_1D = X_Noi.reshape(I * J, )
Ind = np.random.permutation(I * J)
p1 = int(math.floor(I * J * 0.15))
p2 = int(math.floor(I * J * 0.2))
x_Noi_1D[Ind[0:p1]] = x_Noi_1D[Ind[0:p1]] + np.random.randn(p1) * 0.25
# print("All good till Here")
x_Noi_1D[Ind[p1:p1 + p2]] = x_Noi_1D[Ind[p1:p1 + p2]] + (np.random.rand(p2) * 10) - 5  # Add uniform noise
# print("All good till Here")
# test = raw_input()
p3 = int(math.floor(I * J * 0.2))

print("To be changed: {}".format(p1 + p2 + p3))
print("Total numbers is: {}".format(I * J))
x_Noi_1D[Ind[p1 + p2:p1 + p2 + p3]] = x_Noi_1D[Ind[p1 + p2:p1 + p2 + p3]] + (
            np.random.rand(p3) * 4) - 2  # Add uniform noise

# By Caoyuan change all the minus numbers to between 0 and 1
minusId = x_Noi_1D <= 0
minusNum = sum(minusId)
x_Noi_1D[minusId] = np.random.rand(minusNum)
flags = x_Noi_1D <= 0
numFlags = sum(flags)
if numFlags > 0:
    print("Error! there are numbers less than zero==========")

X_Noi = x_Noi_1D.reshape([I, J])
# print("All good till Here")
# test = raw_input()
####Below are initialize the model
lambdaU = np.ones((I, R))
lambdaV = np.ones((J, R))
alphatau, betatau = 3, 1
alpha0, beta0 = 6, 2
maxIter = 300
minIter = 50

a0, b0 = 1e-4, 1e-4
alpha = 1
tol = 1e-5

R5 = 2 * np.ones((I, J))
M = np.ones((I, J))
hyperparams = {'alphatau': alphatau, 'betatau': betatau, 'alpha0': alpha0, 'beta0': beta0, 'alpha': alpha, 'a0': a0,
               'b0': b0, 'maxIter': maxIter, 'minIter': minIter, 'tol': tol}
ARD = True

R = 8
K = 64

model = bnmf_vb(X_Noi, M, R, K, ARD, hyperparams)

model.initialise()
result = model.run(iterations=maxIter)

MSE = model.compute_MSE(M, X_Ori, result)
print("MSE is: {}".format(MSE))

Error = LA.norm((X_Ori - result), 'fro') / LA.norm(X_Ori, 'fro')
print("Error is: {}".format(Error))

# Test R is 2-dim
# with pytest.raises(AssertionError) as error:
#     bnmf_vb(R1, M, R, ARD, hyperparams)
# assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 1-dimensional."
