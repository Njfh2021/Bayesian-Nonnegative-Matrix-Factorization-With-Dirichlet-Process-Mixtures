# project_location = "/path/to/folder/containing/project/"
# import sys
# sys.path.append(project_location) 
import numpy as np
# from code.models.nmf_np import nmf_np
from code.models.bnmf_vb import bnmf_vb

# def __init__(self,R,M,R,ARD,hyperparameters):
I, J, R = 3, 2, 2
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

K = 3
model = bnmf_vb(R5, M, R, K, ARD, hyperparams)

model.initialise()
model.run(iterations=10)

# Test R is 2-dim
# with pytest.raises(AssertionError) as error:
#     bnmf_vb(R1, M, R, ARD, hyperparams)
# assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 1-dimensional."
