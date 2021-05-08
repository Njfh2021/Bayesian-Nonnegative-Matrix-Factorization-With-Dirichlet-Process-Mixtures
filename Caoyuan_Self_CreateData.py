# project_location = "/path/to/folder/containing/project/"
# import sys
# sys.path.append(project_location) 
# from code.models.nmf_np import nmf_np
import math
import numpy as np
import scipy.io as scio

# def __init__(self,R,M,R,ARD,hyperparameters):
I, J, R = 200, 200, 5
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
minusId = x_Noi_1D <= 0
minusNum = sum(minusId)

# By Caoyuan change all the minus numbers to between 0 and 1
x_Noi_1D[minusId] = np.random.rand(minusNum)
# count = 0
# for i in range(I*J):
# 	if x_Noi_1D[i]<0:
# 		count+=1
# print("=============={} numbers less than zero!".format(count))
# test = raw_input()

X_Noi = x_Noi_1D.reshape([I, J])

scio.savemat('X_Ori.mat', {'X_Ori': X_Ori})
scio.savemat('X_Noi.mat', {'X_Noi': X_Noi})
