import numpy as np
import scipy.io as scio
from code1.models.bnmf_vb_ARD_EMG import bnmf_vb_ARD_EMG
from numpy import linalg as LA
allRs = [2,4,5,6]
totalsubject = 1
totalRepe = 1
f = open('RankNOT4','w')

for R in allRs:
    for sub in range(1, totalsubject+1):
        for repe in range(1,totalRepe+1):
        	print("Running R= {}".format(R))
            #print("Rest: {}, repe: {}".format(rest, repe))
            X_Noi = scio.loadmat('EEG/TFFinal.mat')['TFFinal'].astype(float)

            I, J = X_Noi.shape

            alphatau, betatau = 3, 1
            alpha0, beta0 = 6, 2
            maxIter = 300
            minIter = 50

            a0, b0 = 1e-4, 1e-4
            alpha = 1
            tol = 1e-5
            threshold = 1e-3
            #By Caoyuan, set a super big lambda threshold to prevent prune
            thre_Lambda = 5e10

            M = np.ones((I, J))
            hyperparams = {'alphatau': alphatau, 'betatau': betatau, 'alpha0': alpha0, 'beta0': beta0, 'alpha': alpha, 'a0': a0,
                'b0': b0, 'maxIter': maxIter, 'minIter': minIter, 'tol': tol}
            ARD = True
            #Here force the R to be 4
            min_R = R
            max_R = R


            r_array = []
            remain_k_array = []
            K_List = [300]

            #r_total_Error = np.zeros([max_R - min_R + 1, len(K_List)])
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
                        print("Error here!!!")
                        #f.write("Rest {}, repe {}Remaining R is {} \n".format(rest, repe, remain_R))
                        #print("Rest {}, repe {}Remaining R is {} \n".format(rest, repe, remain_R))
                    remain_k_array.append(remain_k)
                    result = np.dot(U, V.T)
                    noise = X_Noi - result
                    #print("After Prune, the rank is: {}".format(remain_R))
                    print('r_{} Finished'.format(r) )
                    scio.savemat('EEG/r_{}.mat'.format(r), {'V':V})
                    #scio.savemat('6moveResult/{}/Vs_{}r_{}.mat'.format(rest,sub, repe), {'V':V})
                    #scio.savemat('Caoyuan_EMGData/V/Vrest{}repe{}.mat'.format(rest, repe), {'V':V})

                    #scio.savemat('Caoyuan_EMGData/result/resultrest{}repe{}R{}.mat'.format(rest, repe,r), {'result':result})
                    #scio.savemat('Caoyuan_EMGData/noise/noiserest{}repe{}R{}.mat'.format(rest, repe, r), {'noise':noise})
f.close()
print('FINISHED')