import numpy as np
from lab1_proto import *
import matplotlib.pyplot as plt
import numpy.testing as npt
#from lab1_data import *

data = np.load('lab1_data.npz', allow_pickle=True)['data']
np.random.shuffle(data)

fig, axs = plt.subplots(len(data[:5]), sharex=True, sharey=True)
fig.suptitle('Utterances')
fig.tight_layout()

for index, item in enumerate(data[:5]):
    print(item)

    result = mfcc(item['samples'])

    print(item['gender'])
    print(item['speaker'])

    axs[index].pcolormesh(result)
    axs[index].set_title(item['gender'] + " " + item['digit'])

plt.show()

