import numpy as np
from lab1_proto import *
import matplotlib.pyplot as plt
import numpy.testing as npt
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import distance
import seaborn as sns

data = np.load('lab1_data.npz', allow_pickle=True)['data']

#res1 = mfcc(data[0]['samples'])
#res2 = mfcc(data[1]['samples'])
#euclidean_dist = np.linalg.norm(res1 - res2)
#print(euclidean_dist)

x = [0, 0, 100, 200, 300, 0, 0]
y = [0, 0, 100, 200, 300, 0, 0]

z = [0, 0, 0, 100, 200, 300, 0]

print(helper_dtw(x, z, distance.euclidean))

mfccs = []
for i in range(len(data)):
    mfccs.append(mfcc(data[i]['samples']))

mfccs = np.concatenate(mfccs)

D = np.zeros([44, 44])

for i in range(44):
    for j in range(44):
        print(str(i) + " " + str(j))
        D[i][j]  = dtw(mfcc(data[i]['samples']),mfcc(data[j]['samples']), distance.euclidean)

sns.reset_defaults()

plt.pcolormesh(D)
plt.colorbar()

fname = 'dtw'

plt.savefig("plots/" + fname + ".png")
plt.show()