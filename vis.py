import os
import sys

import math
import scipy.io as scio

project_location = os.path.dirname(__file__) + "/../../../../"
sys.path.append(project_location)

from exponential import exponential_draw
from normal import normal_draw

import numpy, itertools, os
import numpy as np


import numpy, itertools, matplotlib.pyplot as plt, os


X_Noi = scio.loadmat('Caoyuan_AudioData/MagSTFT.mat')['MagSTFT'].astype(float)
R = X_Noi

print "Mean R: %s. Variance R: %s. Min R: %s. Max R: %s." % (numpy.mean(R),numpy.var(R),R.min(),R.max())
fig = plt.figure()
plt.hist(R.flatten(),bins=range(0,int(R.max())+1))
#plt.hist(R1.flatten(),bins=range(0,int(R1.max())+1))
plt.show()