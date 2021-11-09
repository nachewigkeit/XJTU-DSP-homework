import numpy as np


def averageEnergy(frames):
    energy = np.average(frames * frames, axis=1)
    return energy.reshape(-1)


def zeroCrossingRate(frames):
    _, lens = frames.shape
    delta = np.abs(np.sign(frames[:, 1:]) - np.sign(frames[:, :lens - 1]))
    zeroCrossingRate = np.average(delta / 2, axis=1)
    return zeroCrossingRate.reshape(-1)


def std(frames):
    return np.std(frames, axis=1).reshape(-1)
