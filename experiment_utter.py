import numpy as np
from lab1_proto import *
import matplotlib.pyplot as plt
import numpy.testing as npt
from matplotlib.pyplot import figure
#from lab1_data import *

data = np.load('lab1_data.npz', allow_pickle=True)['data']
data = [item for item in data if item['repetition'] == 'a' and item['gender'] == 'woman' and item['digit'] != 'o' and item['digit'] != 'z']

import matplotlib.pyplot as plt
#plt.figure(figsize=(15,3))

fig, axs = plt.subplots(len(data), figsize=(5,10))
#fig.suptitle('Utterances')
fig.tight_layout()

for i,item in enumerate(data):
    print(item)

    result = mfcc(item['samples'])

    print(item['gender'])
    print(item['speaker'])

    axs[i].pcolormesh(result)
    axs[i].set_title(item['gender'] + " " + item['digit'])

    #plt.title(item['gender'] + " id:" + item['speaker'] + " digit:" + item['digit'])

    #plt.savefig("plots/utter/" + item['gender'] + " " + item['speaker'] + " " + item['digit'] + ".png")

fig.subplots_adjust(hspace=0.5)

# remove the x and y ticks
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

#plt.tight_layout()

fig.savefig("plots/utter/fig_woman.png")
