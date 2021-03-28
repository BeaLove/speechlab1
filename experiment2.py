import numpy as np
from lab1_proto import *
import matplotlib.pyplot as plt

example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()


#int(example['samplingrate'] / 1000) * 2EX0

result = enframe(samples = example['samples'], winlen = 400, winshift = 200)
result2 = preemp(result, p=0.97)

#plt.pcolormesh(result)
plt.pcolormesh(result2)

plt.show()