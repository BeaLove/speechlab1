import numpy as np
from lab1_proto import *
import matplotlib.pyplot as plt
import numpy.testing as npt
#from lab1_data import *

data = np.load('lab1_data.npz', allow_pickle=True)['data']
np.random.shuffle(data)

for index, item in enumerate(data):

    result = mfcc(item['samples'])

    frames.append(result)

    print(item['gender'])
    print(item['speaker'])

    #plt.pcolormesh(result)
    #axs[index].set_title(item['gender'] + " " + item['digit'])

    fname = item['gender'] + " " + item['speaker'] + " " + item['digit']
    plt.pcolormesh(result)
    
    print(result.shape)
    
    break
   #plt.savefig("plots/" + fname + ".png")

