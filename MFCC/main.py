import utils
import matplotlib.pyplot as plt
import numpy as np

signal = utils.get_example(r"1_34_1\0\0_0.wav")
frames = utils.frame(signal, 1000, 400)
frames = utils.window(frames)
pow_frames = utils.get_power(frames)

'''
pre_emphasis = 0.95
emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

plt.figure(figsize=(9, 4))
plt.subplot(211)
plt.plot(range(len(signal)), signal)
plt.subplot(212)
plt.plot(range(len(emphasized_signal)), emphasized_signal)
plt.show()
'''

sample_rate = 44100
nfilt = 40
low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # 将Hz转换为Mel
# 我们要做40个滤波器组，为此需要42个点，这意味着在们需要low_freq_mel和high_freq_mel之间线性间隔40个点
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # 使得Mel scale间距相等
hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # 将Mel转换回-Hz
# bin = sample_rate/NFFT    # frequency bin的计算公式
# bins = hz_points/bin=hz_points*NFFT/ sample_rate    # 得出每个hz_point中有多少frequency bin
bins = np.floor((NFFT + 1) * hz_points / sample_rate)

fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bins[m - 1])  # 左
    f_m = int(bins[m])  # 中
    f_m_plus = int(bins[m + 1])  # 右

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # 数值稳定性
filter_banks = 20 * np.log10(filter_banks)  # dB

print(filter_banks.shape)
