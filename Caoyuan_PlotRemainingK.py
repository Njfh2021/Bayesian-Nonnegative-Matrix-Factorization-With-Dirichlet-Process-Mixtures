import os
import time

import matplotlib
import numpy as np
import scipy.io as scio

matplotlib.use('Agg')
from matplotlib.pyplot import savefig, close
from matplotlib import pyplot as plt

kkError = scio.loadmat('vary_k_result.mat')['vary_k_result'].astype(float)

k = kkError[:, 0].astype(int)
Error = kkError[:, 1]
remainK = kkError[:, 2]
# plot(k,Error)
# plot.show()
result_plot_folder = "fig"

ISOTIMEFORMAT = '%m%d%H%M'
resultFileName = time.strftime(ISOTIMEFORMAT, time.localtime())
# savefig(result_plot_folder+os.sep+resultFileName +'vary_k_result.png')

plt.plot(k, remainK)
my_x_ticks = np.arange(0, 301, 50)
my_y_ticks = np.arange(0, 35, 5)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)

savefig(result_plot_folder + os.sep + resultFileName + 'vary_k_result_remaink.png')
close('all')

# print(r)
# print(Error)
