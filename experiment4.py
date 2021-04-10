import numpy as np
from lab1_proto import *
import matplotlib.pyplot as plt
import numpy.testing as npt
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance

data = np.load('lab1_data.npz', allow_pickle=True)['data']

#res1 = mfcc(data[0]['samples'])
#res2 = mfcc(data[1]['samples'])
#euclidean_dist = np.linalg.norm(res1 - res2)
#print(euclidean_dist)

x = [0, 0, 100, 200, 300, 0, 0]
y = [0, 0, 100, 200, 300, 0, 0]

z = [0, 0, 0, 100, 200, 300, 0]

print(helper_dtw(x, z, distance.euclidean))