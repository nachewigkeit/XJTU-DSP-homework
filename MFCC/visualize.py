import utils
import matplotlib.pyplot as plt
import numpy as np
from python_speech_features import mfcc


def myPlot(x):
    plt.plot(range(len(x)), x)


signal = utils.get_example(r"1_34_1\0\0_0.wav")

'''
origin = utils.frame(signal, 1000, 400)
windowed = utils.window(origin)
originPower = np.log(utils.get_power(origin, NFFT=256))
windowedPower = np.log(utils.get_power(windowed, NFFT=256))
plt.figure(figsize=(10, 5))
plt.subplot(221)
myPlot(origin[0, :])
plt.subplot(222)
myPlot(originPower[0, :])
plt.subplot(223)
myPlot(windowed[0, :])
plt.subplot(224)
myPlot(windowedPower[0, :])
plt.savefig(r"image/window.png", bbox_inches='tight')
'''

plt.figure(figsize=(10, 10))
plt.subplot(211)
myPlot(signal)
plt.subplot(212)
feat = mfcc(signal).T
mul = 8
new = np.zeros((feat.shape[0] * mul, feat.shape[1]))
for i in range(feat.shape[0]):
    new[i * mul:(i + 1) * mul, :] = feat[i, :]
plt.xticks([])
plt.yticks([])
plt.imshow(new, cmap="Blues")
plt.savefig(r"image/mfcc.png", bbox_inches='tight')
