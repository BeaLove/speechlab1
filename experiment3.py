import numpy as np
from lab1_proto import *
import matplotlib.pyplot as plt
import numpy.testing as npt

example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()

#int(example['samplingrate'] / 1000) * 2EX0

print(example['windowed'].shape)

result = enframe(samples = example['samples'], winlen = 400, winshift = 200)
result2 = preemp(result, p=0.97)
result3 = windowing(result2)
result4 = powerSpectrum(result3, 512)
result5 = logMelSpectrum(result4, 20000)

npt.assert_almost_equal(example['frames'], result)
npt.assert_almost_equal(example['preemph'], result2)
npt.assert_almost_equal(example['windowed'], result3)
npt.assert_almost_equal(example['spec'], result4)
npt.assert_almost_equal(example['mspec'], result5)

plt.pcolormesh(result5)

plt.show()