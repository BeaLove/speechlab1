
import numpy as np
from lab1_proto import *
import matplotlib.pyplot as plt
import numpy.testing as npt
#from lab1_data import *
from sklearn.mixture import GaussianMixture


data = np.load('lab1_data.npz', allow_pickle=True)['data']
#data = [item for item in data if item['digit'] == '1']

np.random.shuffle(data)
mfcc_frames = []
mspec_frames = []
labels = []
digits = []

def featureCorrelate(matrix):
    corr = np.corrcoef(matrix, rowvar=False)
    plt.pcolormesh(corr)
    plt.show()

def gaussian(data_concatenacted, mfcc_frames, labels, digits, n_comps=32):
    gmm = GaussianMixture(n_comps)
    gmm.fit(data_concatenacted)
    #predict_labels = gmm.predict(data)
    for idx, point in enumerate(mfcc_frames):
        print(idx)
        print(digits[idx])
        predicted = gmm.predict(point)
        y = np.arange(len(point))
        plt.scatter(y, predicted)
        plt.xlabel("frames")
        plt.ylabel("predicted component")

        title = "predicted phonemes components digit " + str(digits[idx])
        plt.title(title)
        filename = title + "_" + str(idx) + ".png"
        plt.savefig("plots/comp32/" + filename)
        print(filename)
        #print("file saved")
        #plt.show()
    #for i, item in enumerate(predict_labels):
        #print("digit: ", labels[i], "predicted class: ",  item)


def euclideanDistance(digit1, digit2):
    N = digit1.shape[0]
    M = digit2.shape[0]
    distance = np.zeros((N, M))
    for n in range(N):
        for m in range(M):
            distance[n,m] = np.linalg.norm(digit1[n] - digit2[m])
    return distance



for index, item in enumerate(data):

    result = mfcc(item['samples'])

    mfcc_frames.append(result)
    digits.append(item['digit'])

    #result2 = mspec(item['samples'])
    #mspec_frames.append(result2)
    num_frames = len(result)
    label_for_frame = [item['digit']]*len(result)
    labels += label_for_frame

    #print(item['gender'])
    #print(item['speaker'])

    #plt.pcolormesh(result)
    #axs[index].set_title(item['gender'] + " " + item['digit'])

    fname = item['gender'] + " " + item['speaker'] + " " + item['digit']
    #plt.pcolormesh(result)
    
    #print("result", result.shape)
    #print(len(labels))
    #print("result 2", result2.shape)
    
    #break
#print(len(mfcc_frames))
#print(digits)
#print(len(mspec_frames))
mfcc_matrix = np.asarray(mfcc_frames[0])
#mspec_matrix = np.asarray(mspec_frames[0])
for frame in mfcc_frames[1:]:
    mfcc_matrix = np.concatenate((mfcc_matrix, frame), axis=0)
#for frame in mspec_frames[1:]:
 #   mspec_matrix = np.concatenate((mspec_matrix, frame), axis=0)
#print("mfcc",  mfcc_matrix.shape)
#print('mspec', mspec_matrix.shape)

#featureCorrelate(mfcc_matrix)
#featureCorrelate(mspec_matrix)

   #plt.savefig("plots/" + fname + ".png")

gaussian(mfcc_matrix, mfcc_frames, labels, digits, n_comps=4)

#distance_matrix = euclideanDistance(mfcc_frames[0], mfcc_frames[1])




