import scipy.io as scio
ARD20Error=[0.017108182802117219, 0.011477580789724164, 0.015776167674739064, 0.0072219199400691391, 0.005906590991944868, 0.0060859824451229916, 0.0068079133254131318, 0.0064895460994372394, 0.007236871138619161, 0.0067048552671066658, 0.0083708619979113909, 0.0084260154145905514, 0.0084763445818660053, 0.0094606475607714631, 0.010112681852630834, 0.010698955219458784, 0.011069800855436771, 0.010168447512000347, 0.010373137267187657, 0.0098900552938094907]

scio.savemat('r_total_ErrorRARD.mat', {'ARD20Error': ARD20Error})