import numpy as np
import scipy.io as scio
from code1.models.bnmf_vb_ARD_EMG import bnmf_vb_ARD_EMG
from numpy import linalg as LA
singleRest = 0
singleRepe = 0
f = open('RankNOT4','w')

for rest in range(singleRest,singleRest+1):
    for repe in range(singleRepe,singleRepe+1):
        print("Rest: {}, repe: {}".format(rest, repe))
        X_Noi = scio.loadmat('Caoyuan_EmgData/mrest{}repe{}.mat'.format( rest, repe) )['matrix'].astype(float)
        I, J = X_Noi.shape

        alphatau, betatau = 3, 1
        alpha0, beta0 = 6, 2
        maxIter = 300
        minIter = 50

        a0, b0 = 1e-4, 1e-4
        alpha = 1
        tol = 1e-5
        threshold = 1e-3

        thre_Lambda = 5

        M = np.ones((I, J))
        hyperparams = {'alphatau': alphatau, 'betatau': betatau, 'alpha0': alpha0, 'beta0': beta0, 'alpha': alpha, 'a0': a0,
               'b0': b0, 'maxIter': maxIter, 'minIter': minIter, 'tol': tol}
        ARD = True

        min_R = 10
        max_R = 10


        r_array = []
        remain_k_array = []
        K_List = [300]

        r_total_Error = np.zeros([max_R - min_R + 1, len(K_List)])
        count = 0
        for r in range(min_R, max_R + 1):
            k_error_array = []
            for k in K_List:
                #print("r is: {}".format(r))
                r_array.append(r)
                model = bnmf_vb_ARD_EMG(X_Noi, M, r, k, ARD, threshold, thre_Lambda, hyperparams)
                model.initialise()
                U,V, remain_k, remain_R = model.run(iterations=maxIter)
                if remain_R!=4:
                    f.write("Rest {}, repe {}Remaining R is {} \n".format(rest, repe, remain_R))
                    print("Rest {}, repe {}Remaining R is {} \n".format(rest, repe, remain_R))
                remain_k_array.append(remain_k)
                result = np.dot(U, V.T)
                noise = X_Noi - result
                #print("After Prune, the rank is: {}".format(remain_R))
                scio.savemat('Caoyuan_EMGData/U/Urest{}repe{}.mat'.format(rest, repe), {'U':U})
                scio.savemat('Caoyuan_EMGData/V/Vrest{}repe{}.mat'.format(rest, repe), {'V':V})

                scio.savemat('Caoyuan_EMGData/result/resultrest{}repe{}R{}.mat'.format(rest, repe,r), {'result':result})
                scio.savemat('Caoyuan_EMGData/noise/noiserest{}repe{}R{}.mat'.format(rest, repe, r), {'noise':noise})
f.close()