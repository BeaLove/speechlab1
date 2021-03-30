import numpy as np
from lab1_proto import *
import matplotlib.pyplot as plt

example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()

#int(example['samplingrate'] / 1000) * 2EX0

print(example['windowed'].shape)

result = enframe(samples = example['samples'], winlen = 400, winshift = 200)
result2 = preemp(result, p=0.97)
result3 = windowing(result2)
result4 = powerSpectrum(result3, 512)

assert np.all(example['frames']) == np.all(result)
assert np.all(example['preemph']) == np.all(result2)
assert np.all(example['windowed']) == np.all(result3)
assert np.all(example['spec']) == np.all(result3)

#print(example['spec'].shape)
plt.pcolormesh(result4)

plt.show()