import numpy as np
import scipy.io as scio
from code1.models.bnmf_vb_ARD import bnmf_vb_ARD
from numpy import linalg as LA

result = scio.loadmat('Caoyuan_AudioData/resultR20.mat')['result'].astype(float)
X_Ori = scio.loadmat('X_Ori.mat')['X_Ori'].astype(float)

Error = LA.norm((X_Ori - result),'fro')/LA.norm(X_Ori,'fro')
print("Error is: {}".format(Error))