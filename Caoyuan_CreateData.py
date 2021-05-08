"""
Generate a toy dataset for the matrix factorisation case, and store it.

We use dimensions 100 by 50 for the dataset, and 10 latent factors.

As the prior for U and V we take value 1 for all entries (so exp 1).

As a result, each value in R has a value of around 20, and a variance of 100-120.

For contrast, the Sanger dataset of 705 by 140 shifted to nonnegative has mean 
31.522999753779082 and variance 243.2427345740027.

We add Gaussian noise of precision tau = 1 (prior for gamma: alpha=1,beta=1).
(Simply using the expectation of our Gamma distribution over tau)
"""

import os
import sys

import math
import scipy.io as scio

project_location = os.path.dirname(__file__) + "/../../../../"
sys.path.append(project_location)

from exponential import exponential_draw
from normal import normal_draw

import numpy, itertools, matplotlib.pyplot as plt, os
import numpy as np


def generate_dataset(I, J, K, lambdaU, lambdaV, tau):
    # Generate U, V
    U = numpy.zeros((I, K))
    for i, k in itertools.product(xrange(0, I), xrange(0, K)):
        U[i, k] = exponential_draw(lambdaU[i, k])
    V = numpy.zeros((J, K))
    for j, k in itertools.product(xrange(0, J), xrange(0, K)):
        V[j, k] = exponential_draw(lambdaV[j, k])

    # Generate R
    true_R = numpy.dot(U, V.T)
    R = add_noise(true_R, tau)

    return (U, V, tau, true_R, R)


def add_noise(true_R, tau):
    if numpy.isinf(tau):
        return numpy.copy(true_R)

    (I, J) = true_R.shape
    R = numpy.zeros((I, J))
    for i, j in itertools.product(xrange(0, I), xrange(0, J)):
        R[i, j] = normal_draw(true_R[i, j], tau)
    return R


##########

if __name__ == "__main__":
    output_folder = os.path.dirname(__file__) + "/"

    I, J, K = 50, 50, 4  # 20, 10, 5 #
    fraction_unknown = 0.1
    # Add Gaussian noise N(0, 0.5**2)
    alpha, beta = 4., 1.
    lambdaU = numpy.ones((I, K))
    lambdaV = numpy.ones((J, K))
    tau = alpha / beta

    (U, V, tau, true_R, R) = generate_dataset(I, J, K, lambdaU, lambdaV, tau)
    scio.savemat('X_Ori.mat', {'X_Ori': true_R})
    scio.savemat('X_Noi_Gaussian.mat', {'X_Noi_Gaussian': R})

    # By Caoyuan, add sparse Noise Sparse [-5,5]
    X_Noi_Sparse = true_R
    X_Noi_Sparse_1D = X_Noi_Sparse.reshape(I * J, )
    Ind_0 = np.random.permutation(I * J)
    p0 = int(math.floor(I * J * 0.3))
    # Add sparse Noise
    X_Noi_Sparse_1D[Ind_0[0:p0]] = X_Noi_Sparse_1D[Ind_0[0:p0]] + (np.random.rand(p0) * 10) - 5
    X_Noi_Sparse = X_Noi_Sparse_1D.reshape([I, J])
    scio.savemat('X_Noi_Sparse.mat', {'X_Noi_Sparse': X_Noi_Sparse})

    # By Caoyuan, add mixture Noise N(0, 0.5**2), 15%, U[-5,5] 20%, U[-2,2] 20%

    X_Noi_Mixture = true_R
    x_Noi_Mixture_1D = X_Noi_Mixture.reshape(I * J, )
    Ind = np.random.permutation(I * J)

    p1 = int(math.floor(I * J * 0.2))
    x_Noi_Mixture_1D[Ind[0:p1]] = x_Noi_Mixture_1D[Ind[0:p1]] + np.random.randn(p1) * 0.01
    # print("All good till Here")
    p2 = int(math.floor(I * J * 0.2))
    x_Noi_Mixture_1D[Ind[p1:p1 + p2]] = x_Noi_Mixture_1D[Ind[p1:p1 + p2]] + (
                np.random.rand(p2) * 4) - 2  # Add uniform noise
    p3 = int(math.floor(I * J * 0.6))
    x_Noi_Mixture_1D[Ind[p1 + p2:p1 + p2 + p3]] = x_Noi_Mixture_1D[Ind[p1 + p2:p1 + p2 + p3]] + np.random.randn(
        p3) * 0.25

    # (np.random.rand(p3)*4) -2 # Add uniform noise
    X_Noi_Mixture = x_Noi_Mixture_1D.reshape([I, J])
    scio.savemat('X_Noi_Mixture.mat', {'X_Noi_Mixture': X_Noi_Mixture})
    R1 = X_Noi_Mixture

    '''
    # Store all matrices in text files
    numpy.savetxt(open(output_folder+"U.txt",'w'),U)
    numpy.savetxt(open(output_folder+"V.txt",'w'),V)
    numpy.savetxt(open(output_folder+"R_true.txt",'w'),true_R)
    numpy.savetxt(open(output_folder+"R.txt",'w'),R)
    '''

    print
    "Mean R: %s. Variance R: %s. Min R: %s. Max R: %s." % (numpy.mean(R), numpy.var(R), R.min(), R.max())
    fig = plt.figure()
    plt.hist(R.flatten(), bins=range(0, int(R.max()) + 1))
    plt.hist(R1.flatten(), bins=range(0, int(R1.max()) + 1))
    plt.show()
