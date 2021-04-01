import numpy as np
from lab1_proto import *
import matplotlib.pyplot as plt
import numpy.testing as npt
#from lab1_data import *


data = np.load('lab1_data.npz', allow_pickle=True)['data']
np.random.shuffle(data)
mfcc_frames = []
mspec_frames = []
for index, item in enumerate(data):

    result = mfcc(item['samples'])

    mfcc_frames.append(result)

    result2 = mspec(item['samples'])
    mspec_frames.append(result2)

    print(item['gender'])
    print(item['speaker'])

    #plt.pcolormesh(result)
    #axs[index].set_title(item['gender'] + " " + item['digit'])

    fname = item['gender'] + " " + item['speaker'] + " " + item['digit']
    #plt.pcolormesh(result)
    
    print("result", result.shape)
    print("result 2", result2.shape)
    
    #break
print(len(mfcc_frames))
print(len(mspec_frames))
mfcc_matrix = np.asarray(mfcc_frames[0])
mspec_matrix = np.asarray(mspec_frames[0])
for frame in mfcc_frames[1:]:
    mfcc_matrix = np.concatenate((mfcc_matrix, frame), axis=0)
for frame in mspec_frames[1:]:
    mspec_matrix = np.concatenate((mspec_matrix, frame), axis=0)
print("mfcc",  mfcc_matrix.shape)
print('mspec', mspec_matrix.shape)
   #plt.savefig("plots/" + fname + ".png")

