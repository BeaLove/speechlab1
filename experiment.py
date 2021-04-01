import numpy as np
from lab1_proto import *
import matplotlib.pyplot as plt
import numpy.testing as npt
#from lab1_data import *

data = np.load('lab1_data.npz', allow_pickle=True)['data']

for item in data:
    print(item)

    result = mfcc(item['samples'])

    print(item['gender'])
    print(item['speaker'])

    plt.pcolormesh(result)

    plt.show()

    break