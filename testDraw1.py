import matplotlib.pyplot as plt
import numpy as np

p2 = 100
# R = (np.random.rand(p2)*10) -5
R1 = np.random.randn(p2) * 0.25
fig = plt.figure()
# plt.hist(R.flatten(),bins=range(0,int(R.max())+1))
plt.hist(R1.flatten(), bins=range(0, int(R1.max()) + 1))
plt.show()
