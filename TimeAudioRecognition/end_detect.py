import utils
from utils import calEnergy, calZeroCrossingRate, Double_thresh
import numpy as np
import matplotlib.pyplot as plt

path = r"E:\AI\dataset\audio\0.wav"
fig = plt.figure()
# 1. 声音信号解码
info, wave = utils.wavDecode(path)
wave = wave[:, 0]  # wave: [采样次数，]

# 2. 归一化到[-1,1]
wave_max = np.max(np.abs(wave))
wave = wave / wave_max

# x1 = np.arange(5000)
# x2 = np.arange(5000)
# wave1 = np.sin(x1/100)
# wave2 = np.sin(x2/500)
# wave = np.concatenate((wave1,wave2))

# 3. 分帧
frame_lens = 1000
move = 10
frames = utils.frame(wave, 1000, 10)  # frames: [帧数，帧长]

# 4. 短时能量和过零率
energy = calEnergy(frames)  # energy: [帧数]
CrossZeroRate = calZeroCrossingRate(frames)  # CrossZeroRate: [帧数]

# plt.show()
# 5. 双门限检测
frames_num = len(energy)  # 帧数
L, R = Double_thresh(wave, energy, CrossZeroRate)
L_w = L * move + frame_lens // 2
R_w = R * move + frame_lens // 2
# L_w = int(L / frame_lens * len(wave))
# R_w = int(R / frame_lens * len(wave))
# 信号长度 约为 energy长度的 move 倍
x1 = np.arange(len(wave))
x2 = np.arange(len(energy))
fig.add_subplot(311)
plt.scatter([L_w, R_w], [wave[L_w], wave[R_w]], c='r', s=200)
plt.plot(x1, wave)
fig.add_subplot(312)
plt.scatter([L, R], [energy[L], energy[R]], c='r', s=200)
plt.plot(x2, energy)
fig.add_subplot(313)
plt.scatter([L, R], [CrossZeroRate[L], CrossZeroRate[R]], c='r', s=200)
plt.plot(x2, CrossZeroRate)
plt.show()
