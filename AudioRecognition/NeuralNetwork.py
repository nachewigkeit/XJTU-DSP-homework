import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import pickle
from time import time
import numpy as np


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


class AudioDataset(Dataset):
    def __init__(self):
        with open(r"data/x", "rb") as f:
            self.x = pickle.load(f)
        with open(r"data/y", "rb") as f:
            self.y = pickle.load(f)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x = np.reshape(self.x[item], (-1, 16))
        return x, self.y[item]


torch.manual_seed(0)
with open("data/x", "rb") as f:
    x = pickle.load(f)
with open("data/y", "rb") as f:
    y = pickle.load(f)
data = AudioDataset()
length = len(data)
train_size, valid_size = int(0.8 * length), int(0.2 * length)
train_set, valid_set = random_split(data, [train_size, valid_size])
train_loader = DataLoader(dataset=train_set, batch_size=10, shuffle=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=100, shuffle=False)

clf = Classifier(16, 200, 2)

optimizer = opt.Adam(clf.parameters())
scheduler = opt.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
loss_func = nn.CrossEntropyLoss()

logger = SummaryWriter("log/dropout/0.5")

step = 0
for epoch in range(30):
    start = time()
    clf.train()
    for (x, y) in train_loader:
        x_ = Variable(x).to(torch.float32)
        y_ = Variable(y)
        output = clf(x_)
        loss = loss_func(output, y_)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.add_scalar("Loss", loss, step)
        step += 1
    scheduler.step()

    clf.eval()
    with torch.no_grad():
        loss = 0
        acc = 0
        for (x, y) in valid_loader:
            x_ = Variable(x).to(torch.float32)
            y_ = Variable(y)
            output = clf(x_)

            loss += loss_func(output, y_)

            answer = torch.argmax(output, dim=1)
            acc += (answer == y_).sum().float()
        print(epoch)
        print("Loss", loss / len(valid_set))
        print("Acc", acc / len(valid_set))
        print("time:", time() - start)

torch.save(clf.state_dict(), "data/clf.pt")
