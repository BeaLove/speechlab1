import numpy as np
from lab1_proto import *
import matplotlib.pyplot as plt
import numpy.testing as npt
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import distance

data = np.load('lab1_data.npz', allow_pickle=True)['data']
#data = [item for item in data if item['repetition'] == 'a']
print(len(data))
#res1 = mfcc(data[0]['samples'])
#res2 = mfcc(data[1]['samples'])
#euclidean_dist = np.linalg.norm(res1 - res2)
#print(euclidean_dist)

#x = [0, 0, 100, 200, 300, 0, 0]
#y = [0, 0, 100, 200, 300, 0, 0]

#z = [0, 0, 0, 100, 200, 300, 0]

#print(helper_dtw(x, z))

mfccs = []
for i in range(len(data)):
    mfccs.append(mfcc(data[i]['samples']))

mfccs = np.concatenate(mfccs)

D = np.zeros([len(data), len(data)])

for i in range(len(data)):
    for j in range(len(data)):
        print(str(i) + " " + str(j))
        D[i][j]  = dtw(mfcc(data[i]['samples']),mfcc(data[j]['samples']))

plt.pcolormesh(D)
plt.colorbar()

fname = 'dtw7'

plt.savefig("plots/" + fname + ".png")
plt.show()


dendrogram(linkage(D,method='complete'),labels = tidigit2labels(data), orientation='right')
plt.tight_layout()
plt.savefig("plots/dendrogram7.png")

plt.show()