import os
import time

import matplotlib
import numpy as np
import scipy.io as scio

#matplotlib.use('Agg')
from matplotlib.pyplot import savefig
from matplotlib import pyplot as plt

kkError = scio.loadmat('vary_k_result.mat')['vary_k_result'].astype(float)

k = kkError[:, 0].astype(int)
Error = kkError[:, 1]
remainK = kkError[:, 2]
plt.plot(k, Error)

plt.ylabel("Relative Error")
my_x_ticks = np.arange(0, 301, 50)
my_y_ticks = np.arange(3e-2, 7e-2, 8e-3)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.show()
# plot.show()
result_plot_folder = "fig"

ISOTIMEFORMAT = '%m%d%H%M'
resultFileName = time.strftime(ISOTIMEFORMAT, time.localtime())
savefig(result_plot_folder + os.sep + resultFileName + 'vary_k_result.png')

# plot(k,remainK)
# savefig(result_plot_folder+os.sep+resultFileName +'vary_k_result_remaink.png')
# close('all')


# print(r)
# print(Error)
