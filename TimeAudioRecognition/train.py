import torch
import torch.nn as nn
import torch.optim as opt
import torch.utils.data as Data
import pickle
from torch.autograd import Variable
from tqdm import tqdm


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Classifier, self).__init__()
        self.GRU = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, 10)

    def forward(self, x):
        r_out, h_n = self.GRU(x, None)
        out = self.out(r_out[:, -1, :])
        return out


class AudioDataset(Data.Dataset):
    def __init__(self):
        with open(r"data/x", "rb") as f:
            self.x = pickle.load(f)
        with open(r"data/y", "rb") as f:
            self.y = pickle.load(f)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


clf = Classifier(1, 5, 1)
train_data = AudioDataset()
train_loader = Data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
optimizer = opt.Adam(clf.parameters())
loss_func = nn.CrossEntropyLoss()
for epoch in range(1):
    for (x, y) in tqdm(train_loader):
        x_ = Variable(x).to(torch.float32)
        y_ = Variable(y)
        output = clf(x_)
        loss = loss_func(output, y_)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        sum = 0
        for (x, y) in tqdm(train_loader):
            x_ = Variable(x).to(torch.float32)
            y_ = Variable(y)
            output = clf(x_)
            loss = loss_func(output, y_)
            sum += loss
        print(sum / len(train_data))
