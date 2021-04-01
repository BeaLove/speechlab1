import numpy as np
from lab1_proto import *
import matplotlib.pyplot as plt

example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()

#int(example['samplingrate'] / 1000) * 2EX0

print(example['windowed'].shape)

result = enframe(samples = example['samples'], winlen = 400, winshift = 200)
result2 = preemp(result, p=0.97) #signal.lfilter([1, 0.97],[1], example['frames']) #
result3 = windowing(result2)
result4 = powerSpectrum(result3, 512)


assert np.array_equal(example['frames'], result)
assert np.array_equal(example['preemph'], result2)
assert np.array_equal(example['windowed'], result3)
assert np.array_equal(example['spec'], result4)

plt.pcolormesh(result4)

plt.show()