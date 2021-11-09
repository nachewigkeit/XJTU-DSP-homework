import os
import config
import utils
from tqdm import tqdm
import pickle

wlen = 512  # 帧长
inc = 128  # 帧移

datadict = {}

for group in tqdm(os.listdir(config.datasetPath)):
    for num in range(10):
        numdir = os.path.join(config.datasetPath, group, str(num))
        for file in os.listdir(numdir):
            # 读wav文件并归一化
            info, wave_data = utils.wavDecode(os.path.join(numdir, file))
            wave_data = wave_data[:, 0] * 1.0 / (max(abs(wave_data[:, 0])))
            framerate = info[2]

            # 帧数
            frames = utils.split(wave_data, 100)
            nf = frames.shape[0]

            energy = utils.calEnergy(frames).reshape((-1, 1))

            if num not in datadict.keys():
                datadict[num] = []
            datadict[num].append(energy)

x = []
y = []

for num in range(10):
    for data in datadict[num]:
        x.append(data)
        y.append(num)
print(len(x), len(y))

with open(r"data/x", "wb") as f:
    pickle.dump(x, f)
with open(r"data/y", "wb") as f:
    pickle.dump(y, f)
