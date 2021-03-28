import numpy as np
from lab1_proto import *
import matplotlib.pyplot as plt

example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()

#int(example['samplingrate'] / 1000) * 2EX0

result = enframe(samples = example['samples'], winlen = 400, winshift = 200)

print(len(result))
print(len(example['frames']))


print(np.array(result).shape)
print(np.array(example['frames']).shape)

print(len(example['samples']))

plt.pcolormesh(result)

plt.show()