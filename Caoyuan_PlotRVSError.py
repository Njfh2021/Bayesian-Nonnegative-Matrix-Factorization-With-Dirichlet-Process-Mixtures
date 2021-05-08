import os
import time

import matplotlib
import numpy as np
import scipy.io as scio

matplotlib.use('Agg')
# from matplotlib.pyplot import plot,savefig, close
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig, close

rError = scio.loadmat('vary_r_result.mat')['vary_r_result'].astype(float)

r = rError[:, 0].astype(int)
Error = rError[:, 1]
plt.plot(r, Error)
plt.ylabel("Relative Error")
my_x_ticks = np.arange(0, 25, 5)
my_y_ticks = np.arange(3e-2, 7e-2, 8e-3)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.show()
ISOTIMEFORMAT = '%m%d%H%M'
resultFileName = time.strftime(ISOTIMEFORMAT, time.localtime())

result_plot_folder = "fig"
savefig(result_plot_folder + os.sep + resultFileName + 'vary_r_result.png')
close('all')

# print(r)
# print(Error)
