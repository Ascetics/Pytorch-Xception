"""
按照论文《Xception: Deep Learning with Depthwise Separable Convolutions》实现

摘自论文的Figure 5的一段话
The Xception architecture:
the data first goes through the entry flow, then through the middle flow which
is repeated eight times, and finally through the exit flow. Note that all
Convolution and SeparableConvolution layers are followed by batch normalization
[7] (not included in the diagram). All SeparableConvolution layers use a depth
multiplier of 1 (no depth expansion).


实现方式：
论文说Xception共36个Conv layer，组成14个module。为了方便，我们以module为单位实现。

对EntryFlow：
第1个module，也就是普通卷积单独实现。
第2-第4个module结构相似，都是下采样，用DownBlock实现。

对MiddleFlow：
第5-第12个module结构相似，都不进行下采样，用Block实现。

对ExitFlow：
第14个module有下采样，通道数不同，用ExitDownBlock实现。
第15个module拆开，卷积部分单独实现

全局平均池化单独实现。后面接全连接层。

读论文不易，转载请注明github地址
https://github.com/Ascetics/Pytorch-Xception/tree/master
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True):
        """
        深度可分离卷积
        第一个卷积在spatial层面上，每个channel单独进行卷积，用group=out_channels实现
        第二个卷积在cross-channel层面上，相当于用1x1卷积调整维度
        :param in_channels: 输入channels
        :param out_channels: 输出channels
        :param kernel_size: depthwise conv的kernel_size
        :param stride: depthwise conv的stride
        :param padding: depthwise conv的padding
        :param bias: 两个卷积的bias
        """
        super(SeparableConv2d, self).__init__()
        self.dconv = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                               padding, groups=in_channels, bias=bias)
        self.pconv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               bias=bias)
        pass

    def forward(self, x):
        return self.pconv(self.dconv(x))  # 先depthwise conv，后pointwise conv

    pass


class ResidualConnection(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        论文中的Residual Connection，来自ResNet的启发，也叫project、skip等
        调整维数
        :param in_channels: 输入channels
        :param out_channels: 输出channels
        :param stride: 下采样，默认不下采样，只调整channel
        """
        super(ResidualConnection, self).__init__(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        pass

    pass


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, relu1=True):
        """
        Entry Flow的3个下采样module
        按论文所说，每个Conv和Separable Conv都需要跟BN
        论文Figure 5中，第1个Separable Conv前面没有ReLU，需要判断一下，
        论文Figure 5中，每个module的Separable Conv的out_channels一样，MaxPool做下采样
        :param in_channels: 输入channels
        :param out_channels: 输出channels
        :param relu1: 判断有没有第一个ReLU，默认是有的
        """
        super(DownBlock, self).__init__()
        self.project = ResidualConnection(in_channels, out_channels, stride=2)
        self.relu1 = None
        if relu1:
            self.relu1 = nn.ReLU(inplace=True)
        self.sepconv1 = SeparableConv2d(in_channels, out_channels,
                                        kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu2 = nn.ReLU(inplace=True)
        self.sepconv2 = SeparableConv2d(out_channels, out_channels,
                                        kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        pass

    def forward(self, x):
        identity = self.project(x)  # residual connection 准备

        if self.relu1:  # 第1个Separable Conv前面没有ReLU，需要判断一下
            x = self.relu1(x)
        x = self.sepconv1(x)  # 第2个Separable Conv
        x = self.bn1(x)

        x = self.relu2(x)
        x = self.sepconv2(x)  # 第2个Separable Conv
        x = self.bn2(x)

        x = self.maxpool(x)  # 下采样2倍

        x += identity  # residual connection 相加
        return x

    pass


class Block(nn.Module):
    def __init__(self, inplanes=728):
        """
        Middle Flow中重复的block，channels和spatial都不发生变化
        :param inplanes:
        """
        super(Block, self).__init__()
        mods = [nn.ReLU(inplace=True),
                SeparableConv2d(inplanes, inplanes, 3, padding=1, bias=False),
                nn.BatchNorm2d(inplanes)]
        mods *= 3  # 重复3次ReLU、SeparableConv、BN
        self.convs = nn.Sequential(*mods)
        pass

    def forward(self, x):
        return x + self.convs(x)  # channels和spatial都没有发生变化，直接相加

    pass


class ExitDownBlock(nn.Module):
    def __init__(self, in_channels=728, out_channels=1024):
        """
        Exit Flow的第1个module
        两个Separable Conv的输出channels不一样，都不做下采样
        Maxpool下采样2倍
        :param in_channels: 输入channels
        :param out_channels: 输出channels
        """
        super(ExitDownBlock, self).__init__()
        self.project = ResidualConnection(in_channels, out_channels, stride=2)

        self.relu1 = nn.ReLU(inplace=True)
        self.sepconv1 = SeparableConv2d(in_channels, in_channels,
                                        kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.relu2 = nn.ReLU(inplace=True)
        self.sepconv2 = SeparableConv2d(in_channels, out_channels,
                                        kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        pass

    def forward(self, x):
        identity = self.project(x)  # residual connection 准备

        x = self.relu1(x)
        x = self.sepconv1(x)  # 第1个Separable Conv
        x = self.bn1(x)

        x = self.relu2(x)
        x = self.sepconv2(x)  # 第2个Separable Conv
        x = self.bn2(x)

        x = self.maxpool(x)  # 下采样2倍

        x += identity  # residual connection 相加
        return x

    pass


class Xception(nn.Module):
    def __init__(self, in_channels=3, n_class=1000):
        super(Xception, self).__init__()
        # 以下Entry Flow
        conv1 = [nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
                 nn.BatchNorm2d(32),
                 nn.ReLU(inplace=True), ]
        self.entry_conv1 = nn.Sequential(*conv1)

        conv2 = [nn.Conv2d(32, 64, 3, padding=1, bias=False),
                 nn.BatchNorm2d(64),
                 nn.ReLU(inplace=True), ]
        self.entry_conv2 = nn.Sequential(*conv2)

        self.entry_block1 = DownBlock(64, 128, relu1=False)
        self.entry_block2 = DownBlock(128, 256)
        self.entry_block3 = DownBlock(256, 728)

        # 以下Middle Flow
        self.middle_flow = nn.ModuleList([Block(728)] * 8)

        # 以下Exit Flow
        self.exit_block = ExitDownBlock(728, 1024)

        conv1 = [nn.Conv2d(1024, 1536, 3, padding=1, bias=False),
                 nn.BatchNorm2d(1536),
                 nn.ReLU(inplace=True), ]
        self.exit_conv1 = nn.Sequential(*conv1)

        conv2 = [nn.Conv2d(1536, 2048, 3, padding=1, bias=False),
                 nn.BatchNorm2d(2048),
                 nn.ReLU(inplace=True), ]
        self.exit_conv2 = nn.Sequential(*conv2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, n_class)

        pass

    def forward(self, x):
        # Entry Flow

        x = self.entry_conv1(x)
        x = self.entry_conv2(x)

        x = self.entry_block1(x)
        x = self.entry_block2(x)
        x = self.entry_block3(x)

        # Middle Flow
        for block in self.middle_flow:
            x = block(x)

        # Exit Flow
        x = self.exit_block(x)
        x = self.exit_conv1(x)
        x = self.exit_conv2(x)

        # FC
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

    pass


if __name__ == '__main__':
    # device = torch.device('cuda:6')
    device = torch.device('cpu')

    net = Xception(in_channels=3, n_class=8)
    print('in:', net)
    net.to(device)

    in_data = torch.randint(0, 256, (24, 3, 299, 299), dtype=torch.float)
    print(in_data.shape)
    in_data = in_data.to(device)

    out_data = net(in_data)
    out_data = out_data.cpu()
    print('out:', out_data.shape)
    pass
