import utils
import matplotlib.pyplot as plt

path = r"E:\dataset\audio\2_34_1\2\2193311468-1.wav"
info, wave_data = utils.wavDecode(path)
wave_data = wave_data[:, 0]
wave_data = wave_data * 1.0 / (max(abs(wave_data)))
plt.plot(range(len(wave_data)), wave_data)
plt.show()
