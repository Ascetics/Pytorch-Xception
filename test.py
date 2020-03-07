import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class Wrong(nn.Module):
    def __init__(self):
        super(Wrong, self).__init__()
        self.convs = nn.Sequential(nn.ReLU(inplace=True),
                                   nn.Conv2d(3, 3, 3, padding=1))
        self.residual = nn.Conv2d(3, 3, 3, padding=1)
        pass

    def forward(self, x):
        r = self.residual(x)  # 卷积之后，x就不能modify了
        h = self.convs(x)  # relu就modify了x，反向传播时候会报错
        h = h + r
        return h

    pass


class RightOne(nn.Module):
    def __init__(self):
        super(RightOne, self).__init__()
        self.convs = nn.Sequential(nn.ReLU(inplace=False),  # 改法1，别省内存了
                                   nn.Conv2d(3, 3, 3, padding=1))
        self.residual = nn.Conv2d(3, 3, 3, padding=1)
        pass

    def forward(self, x):
        r = self.residual(x)
        h = self.convs(x)
        h = h + r
        return h

    pass


class RightTwo(nn.Module):
    def __init__(self):
        super(RightTwo, self).__init__()
        self.convs = nn.Sequential(nn.ReLU(inplace=True),
                                   nn.Conv2d(3, 3, 3, padding=1))
        self.residual = nn.Conv2d(3, 3, 3, padding=1)
        pass

    def forward(self, x):
        r = self.residual(x.clone())  # 改法2，clone还是消耗内存的
        h = self.convs(x)
        h = h + r
        return h

    pass


if __name__ == '__main__':
    in_data = torch.randint(-2, 2, (1, 3, 2, 2), dtype=torch.float)
    in_label = torch.randint(0, 3, (1, 2, 2))

    print(in_data.shape)

    func = nn.CrossEntropyLoss()
    t = RightTwo()

    in_data = in_data.cuda()
    in_label = in_label.cuda()
    t.cuda()
    out_data = t(in_data)
    print(out_data.shape)

    loss = func(out_data, in_label)
    loss.backward()
