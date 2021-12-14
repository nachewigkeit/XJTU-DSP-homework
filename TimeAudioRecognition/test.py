import numpy as np
import torch
import torch.nn as nn
import utils
import feature


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Classifier, self).__init__()
        self.GRU = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=True,
                          dropout=0.5,
                          batch_first=True)
        self.out = nn.Linear(2 * hidden_size, 10)

    def forward(self, x):
        r_out, h_n = self.GRU(x, None)
        out = self.out(r_out[:, -1, :])
        return out


clf = Classifier(16, 200, 2)
clf.load_state_dict(torch.load("data/clf.pt"))

info, wave_data = utils.wavDecode(r"E:\AI\dataset\audio\1_34_1\0\0_1.wav")
wave_data = wave_data[:, 0]
wave_data = wave_data * 1.0 / (max(abs(wave_data)))
wave_data = utils.double_thresh(wave_data)
feat = feature.compute_mfcc(wave_data, numcep=16, nfilt=20, split=10)
x = np.reshape(feat, (1, -1, 16))
x = torch.tensor(x)
x_ = torch.autograd.Variable(x).to(torch.float32)
y = clf(x_)
y = torch.softmax(y, -1)
print(y)
