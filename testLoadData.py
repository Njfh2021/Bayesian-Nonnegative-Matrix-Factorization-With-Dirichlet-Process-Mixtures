import scipy.io as scio

relation_matrix = scio.loadmat('MagSTFT.mat')['MagSTFT'].astype(float)
print(relation_matrix)
