import utils
import config
import os
import feature
import numpy as np
import matplotlib.pyplot as plt

info, wave_data = utils.wavDecode(os.path.join(config.datasetPath, r"1_34_1\0\0_0.wav"))
wave_data = wave_data[:, 0]
wave_data = wave_data * 1.0 / (max(abs(wave_data)))

frame_lens = 512
move = 128

frames = utils.frame(wave_data, frame_lens, move)
energy = feature.averageEnergy(frames)
CrossZeroRate = feature.zeroCrossingRate(frames)

energy_mean = energy.mean()
T1 = np.mean(energy[:10])
T2 = energy_mean / 4  # 较高的能量阈值
T1 = (T1 + T2) / 4  # 较低的能量阈值

range_o = np.arange(len(energy))
# 首先利用较大能量阈值 MH 进行初步检测
mask1 = energy > T2
range1 = range_o[mask1]
N3, N4 = range1[0], range1[-1]

# 利用较小能量阈值 ML 进行第二步能量检测
N2, N5 = N3, N4
for i in range_o[:N3][::-1]:  # 从N3向左搜索 从N4向右搜索
    if energy[i] <= T1:
        N2 = i
        break
for j in range_o[N4:]:
    if energy[j] <= T1:
        N5 = j
        break
L = N2
R = N5
L_w = N2 * move + frame_lens // 2
R_w = N5 * move + frame_lens // 2

fig = plt.figure(figsize=(9, 6))
x2 = np.arange(len(energy))
x3 = np.arange(len(wave_data))

fig.add_subplot(311)
plt.title("Wave")
plt.xticks([])
plt.ylim([wave_data.min(), wave_data.max()])
plt.plot(x3, wave_data)
plt.plot([L_w, L_w], [wave_data.min(), wave_data.max()], c='r', linestyle='--')
plt.plot([R_w, R_w], [wave_data.min(), wave_data.max()], c='r', linestyle='--')

fig.add_subplot(312)
plt.title("Energy")
plt.xticks([])
plt.ylim([energy.min(), energy.max()])
plt.plot(x2, energy)
plt.plot([L, L], [energy.min(), energy.max()], c='r', linestyle='--')
plt.plot([R, R], [energy.min(), energy.max()], c='r', linestyle='--')

fig.add_subplot(313)
plt.title("CrossZeroRate")
plt.xticks([])
plt.ylim([CrossZeroRate.min(), CrossZeroRate.max()])
plt.plot(x2, CrossZeroRate)
plt.plot([L, L], [CrossZeroRate.min(), CrossZeroRate.max()], c='r', linestyle='--')
plt.plot([R, R], [CrossZeroRate.min(), CrossZeroRate.max()], c='r', linestyle='--')

plt.savefig(r"image/double_thres.png", bbox_inches='tight')
