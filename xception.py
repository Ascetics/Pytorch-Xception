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


"""
论文《Xception: Deep Learning with Depthwise Separable Convolutions》

摘自Xception论文的Figure 5的一段话
The Xception architecture:
the data first goes through the entry flow, then through the middle flow which
is repeated eight times, and finally through the exit flow. Note that all
Convolution and SeparableConvolution layers are followed by batch normalization
[7] (not included in the diagram). All SeparableConvolution layers use a depth
multiplier of 1 (no depth expansion).


按照Xception论文实现方式：
论文说Xception共36个Conv layer，组成14个module。为了方便，我们以module为单位实现。

对EntryFlow：
第1个module，也就是普通卷积单独实现。
第2-第4个module结构相似，都是下采样，用_PoolEntryBlock实现。

对MiddleFlow：
第5-第12个module结构相似，都不进行下采样，用_PoolMiddleBlock实现。

对ExitFlow：
第14个module有下采样，通道数不同，用_PoolExitBlock实现。
第15个module拆开，卷积部分单独实现

全局平均池化单独实现。后面接全连接层。

读论文不易，转载请注明github地址
https://github.com/Ascetics/Pytorch-Xception/tree/master
"""


class _PoolEntryBlock(nn.Module):
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
        super(_PoolEntryBlock, self).__init__()
        self.project = ResidualConnection(in_channels, out_channels, stride=2)
        self.relu1 = None
        if relu1:
            self.relu1 = nn.ReLU(inplace=False)
            # self.relu1的inplace必须是False，否则loss.backward()会报错
            # self.project做ResidualConnection是卷积操作，要求x不能被modify。inplace=True就modify了x
            # 同理，后面两处inplace也必须是False，因为他们都是Block的第一个操作
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

        x = x + identity  # residual connection 相加
        return x

    pass


class _PoolMiddleBlock(nn.Module):
    def __init__(self, inplanes=728):
        """
        Middle Flow中重复的block，channels和spatial都不发生变化
        :param inplanes: 输入channels
        """
        super(_PoolMiddleBlock, self).__init__()
        mods = [nn.ReLU(inplace=False),  # 必须是False，否则loss.backward()会报错
                SeparableConv2d(inplanes, inplanes, 3, padding=1, bias=False),
                nn.BatchNorm2d(inplanes)]
        mods *= 3  # 重复3次ReLU、SeparableConv、BN
        self.convs = nn.Sequential(*mods)
        pass

    def forward(self, x):
        x + self.convs(x)  # channels和spatial都没有发生变化，直接相加
        return x

    pass


class _PoolExitBlock(nn.Module):
    def __init__(self, in_channels=728, out_channels=1024):
        """
        Exit Flow的第1个module
        两个Separable Conv的输出channels不一样，都不做下采样
        Maxpool下采样2倍
        :param in_channels: 输入channels
        :param out_channels: 输出channels
        """
        super(_PoolExitBlock, self).__init__()
        self.project = ResidualConnection(in_channels, out_channels, stride=2)

        self.relu1 = nn.ReLU(inplace=False)  # 必须是False，否则loss.backward()会报错
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

        x = x + identity  # residual connection 相加
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

        self.entry_block1 = _PoolEntryBlock(64, 128, relu1=False)
        self.entry_block2 = _PoolEntryBlock(128, 256)
        self.entry_block3 = _PoolEntryBlock(256, 728)

        # 以下Middle Flow
        self.middle_flow = nn.ModuleList([_PoolMiddleBlock(728)] * 8)

        # 以下Exit Flow
        self.exit_block = _PoolExitBlock(728, 1024)

        conv1 = [SeparableConv2d(1024, 1536, 3, padding=1, bias=False),
                 # nn.Conv2d(1024, 1536, 3, padding=1, bias=False),
                 nn.BatchNorm2d(1536),
                 nn.ReLU(inplace=True), ]
        self.exit_conv1 = nn.Sequential(*conv1)

        conv2 = [SeparableConv2d(1536, 2048, 3, padding=1, bias=False),
                 # nn.Conv2d(1536, 2048, 3, padding=1, bias=False),
                 nn.BatchNorm2d(2048),
                 nn.ReLU(inplace=True), ]
        self.exit_conv2 = nn.Sequential(*conv2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(2048, n_class),
                                nn.ReLU(inplace=True))

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


"""
论文《Deeplab v3+：Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation》

摘自DeepLabV3+论文 Fig 4 的一段话
Fig. 4. We modify the Xception as follows: 
(1) more layers (same as MSRA's modifcation except the changes in Entry flow).
(2) all the max pooling operations are replaced by depthwise separable convolutions with striding.
(3) extra batch normalization and ReLU are added after each 3x3 depthwise convolution, similar to MobileNet.

按照DeepLabV3+论文中对Xception的改进实现XceptionBackbone，为DeepLabV+做准备：

对EntryFlow：
第1个module，也就是普通卷积单独实现。
第2-第4个module结构相似，都是stride=2的SeprableConv下采样，用_ComvEntryBlock实现。

对MiddleFlow：
第5-第12个module结构相似，都不进行下采样，用_ConvMiddleBlock实现。

对ExitFlow：
第14个module有stride=2的SeprableConv下采样，通道数不同，用_ConvExitBlock实现。
第15个module拆开，3个卷积单独实现

全局平均池化单独实现。后面接全连接层。
"""


class _ConvEntryBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Entry Flow的3个下采样module
        按论文所说，每个Conv和Separable Conv都需要跟BN，ReLU
        每个module的Separable Conv的out_channels一样，stride=2的SeprableConv做下采样
        :param in_channels: 输入channels
        :param out_channels: 输出channels
        """
        super(_ConvEntryBlock, self).__init__()
        self.project = ResidualConnection(in_channels, out_channels, stride=2)
        convs = [SeparableConv2d(in_channels, out_channels, 3, padding=1,  # 第1个SeparableConv2d,不下采样
                                 bias=False),
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(inplace=True),
                 SeparableConv2d(out_channels, out_channels, 3, padding=1,  # 第2个SeparableConv2d,不下采样
                                 bias=False),
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(inplace=True),
                 SeparableConv2d(out_channels, out_channels, 3, stride=2,  # 第2个SeparableConv2d,stride=2,下采样2倍
                                 padding=1, bias=False),

                 nn.BatchNorm2d(out_channels), ]
        self.convs = nn.Sequential(*convs)
        pass

    def forward(self, x):
        identity = self.project(x)  # residual connection 准备
        x = self.convs(x)  # 下采样2倍
        x = x + identity  # residual connection 相加
        return F.relu(x, inplace=True)

    pass


class _ConvMiddleBlock(nn.Module):
    def __init__(self, inplanes=728):
        """
        Middle Flow中重复的block，channels和spatial都不发生变化
        :param inplanes: 输入channels
        """
        super(_ConvMiddleBlock, self).__init__()
        convs = [SeparableConv2d(inplanes, inplanes, 3, padding=1, bias=False),
                 nn.BatchNorm2d(inplanes),
                 nn.ReLU(inplace=True),
                 SeparableConv2d(inplanes, inplanes, 3, padding=1, bias=False),
                 nn.BatchNorm2d(inplanes),
                 nn.ReLU(inplace=True),
                 SeparableConv2d(inplanes, inplanes, 3, padding=1, bias=False),
                 nn.BatchNorm2d(inplanes), ]
        self.convs = nn.Sequential(*convs)
        pass

    def forward(self, x):
        x = x + self.convs(x)  # channels和spatial都没有发生变化，Residual Connection直接相加
        return F.relu(x, inplace=True)

    pass


class _ConvExitBlock(nn.Module):
    def __init__(self, in_channels=728, out_channels=1024):
        """
        Exit Flow的第1个module
        前两个Separable Conv都不做下采样
        最后一个Separable Conv下采样2倍
        :param in_channels: 输入channels
        :param out_channels: 输出channels
        """
        super(_ConvExitBlock, self).__init__()
        self.project = ResidualConnection(in_channels, out_channels, stride=2)
        convs = [SeparableConv2d(in_channels, in_channels, 3, padding=1,
                                 bias=False),  # 728->728，不下采样
                 nn.BatchNorm2d(in_channels),
                 nn.ReLU(inplace=True),
                 SeparableConv2d(in_channels, out_channels, 3, padding=1,
                                 bias=False),  # 728->1024，不下采样
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(inplace=True),
                 SeparableConv2d(out_channels, out_channels, 3, stride=2,
                                 padding=1, bias=False),  # 1024->1024，下采样2倍
                 nn.BatchNorm2d(out_channels), ]
        self.convs = nn.Sequential(*convs)
        pass

    def forward(self, x):
        identity = self.project(x)  # residual connection 准备
        x = self.convs(x)  # 下采样2倍
        x = x + identity  # residual connection 相加
        return F.relu(x, inplace=True)

    pass


class XceptionBackbone(nn.Module):
    def __init__(self, in_channels=3, n_class=1000):
        super(XceptionBackbone, self).__init__()
        # 以下Entry Flow
        conv1 = [nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
                 nn.BatchNorm2d(32),
                 nn.ReLU(inplace=True), ]
        self.entry_conv1 = nn.Sequential(*conv1)

        conv2 = [nn.Conv2d(32, 64, 3, padding=1, bias=False),
                 nn.BatchNorm2d(64),
                 nn.ReLU(inplace=True), ]
        self.entry_conv2 = nn.Sequential(*conv2)

        self.entry_block1 = _ConvEntryBlock(64, 128)
        self.entry_block2 = _ConvEntryBlock(128, 256)
        self.entry_block3 = _ConvEntryBlock(256, 728)

        # 以下Middle Flow
        self.middle_flow = nn.ModuleList([_ConvMiddleBlock(728)] * 16)  # 改进之一，middle block有16个

        # 以下Exit Flow
        self.exit_block = _ConvExitBlock(728, 1024)

        conv1 = [SeparableConv2d(1024, 1536, 3, padding=1, bias=False),
                 nn.BatchNorm2d(1536),
                 nn.ReLU(inplace=True), ]
        self.exit_conv1 = nn.Sequential(*conv1)

        conv2 = [SeparableConv2d(1536, 1536, 3, padding=1, bias=False),
                 nn.BatchNorm2d(1536),
                 nn.ReLU(inplace=True), ]
        self.exit_conv2 = nn.Sequential(*conv2)

        conv3 = [SeparableConv2d(1536, 2048, 3, padding=1, bias=False),
                 nn.BatchNorm2d(2048),
                 nn.ReLU(inplace=True), ]
        self.exit_conv3 = nn.Sequential(*conv3)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(2048, n_class),
                                nn.ReLU(inplace=True))

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
        x = self.exit_conv3(x)

        # FC
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

    pass


################################################################################

class _XceptionFactory(nn.Module):
    def __init__(self, in_channels,
                 entry_block=_PoolEntryBlock, entry_channels=(128, 256, 728),
                 middle_block=_PoolMiddleBlock, n_middle=8,
                 exit_block=_PoolExitBlock, exit_channels=1024,
                 exit_conv_channels=(1536, 2048),
                 n_class=1000):
        """
        闲的蛋疼，实现一个工厂类，将Xception论文实现和DeepLabV+论文改进的XceptionBackbone
        统一到一起。增强了扩展性。
        :param in_channels: 输入channels，图像channels
        :param entry_block: 论文实现用_PoolEntryBlock，改进实现用_ConvEntryBlock
        :param entry_channels: 每个entry block的输出channels，列表类型
        :param middle_block: 论文实现用_PoolMiddleBlock，改进实现用_ConvMiddleBlock
        :param n_middle: 论文实现用8，改进实现用16
        :param exit_block: 论文实现用_PoolExitBlock，改进实现用_ConvExitBlock
        :param exit_channels: 论文实现、改进实现都是1024
        :param exit_conv_channels: 论文实现是2个Separable Conv的输出channels(1536,,2048)；
                                    改进实现是3个Separable Conv的输出channels(1536,1536,2048)
        :param n_class: n种分类
        """
        super(_XceptionFactory, self).__init__()
        # 以下Entry Flow
        conv1 = [nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
                 nn.BatchNorm2d(32),
                 nn.ReLU(inplace=True), ]
        self.entry_conv1 = nn.Sequential(*conv1)  # 第1个普通卷积，下采样2倍

        conv2 = [nn.Conv2d(32, 64, 3, padding=1, bias=False),
                 nn.BatchNorm2d(64),
                 nn.ReLU(inplace=True), ]
        self.entry_conv2 = nn.Sequential(*conv2)  # 第2个普通卷积

        self.entry_blocks = nn.ModuleList()  # 连续3个residual block，都下采样2倍。
        in_channels = 64
        for i, out_channels in enumerate(entry_channels):
            if i == 0 and isinstance(entry_block, _PoolEntryBlock):  # 注意第一个residual block的relu不一样
                self.entry_blocks.append(entry_block(in_channels, out_channels, relu1=False))
            elif i == 0 and isinstance(entry_block, _ConvEntryBlock):
                self.entry_blocks.append(entry_block(in_channels, out_channels))
            else:
                self.entry_blocks.append(entry_block(in_channels, out_channels))
            in_channels = out_channels
            pass

        # 以下Middle Flow
        self.middle_blocks = nn.ModuleList([middle_block(in_channels)] * n_middle)

        # 以下Exit Flow
        self.exit_block = exit_block(in_channels, exit_channels)
        in_channels = exit_channels

        self.exit_convs = nn.ModuleList()
        for out_channels in exit_conv_channels:
            conv = [SeparableConv2d(in_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True), ]
            self.exit_convs.append(nn.Sequential(*conv))
            in_channels = out_channels
            pass

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_channels, n_class),
                                nn.ReLU(inplace=True))
        pass

    def forward(self, x):
        # Entry Flow
        x = self.entry_conv1(x)
        x = self.entry_conv2(x)
        for block in self.entry_blocks:
            x = block(x)
            pass

        # Middle Flow
        for block in self.middle_blocks:
            x = block(x)
            pass

        # Exit Flow
        x = self.exit_block(x)

        for conv in self.exit_convs:
            x = conv(x)
            pass

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

    pass


def xception(xception_type='backbone', in_channels=3, n_class=1000):
    if xception_type == 'paper':
        return _XceptionFactory(in_channels=in_channels,
                                entry_block=_PoolEntryBlock, entry_channels=[128, 256, 728],
                                middle_block=_PoolMiddleBlock, n_middle=8,
                                exit_block=_PoolExitBlock, exit_channels=1024,
                                exit_conv_channels=[1536, 2048],
                                n_class=n_class)
    elif xception_type == 'backbone':
        return _XceptionFactory(in_channels=in_channels,
                                entry_block=_ConvEntryBlock, entry_channels=[128, 256, 728],
                                middle_block=_ConvMiddleBlock, n_middle=16,
                                exit_block=_ConvExitBlock, exit_channels=1024,
                                exit_conv_channels=[1536, 1536, 2048],
                                n_class=n_class)
    else:
        raise ValueError('xception type error!')


################################################################################

if __name__ == '__main__':
    # device = torch.device('cuda:6')
    device = torch.device('cpu')

    # net = Xception(3, 8).to(device)
    # net = XceptionBackbone(3, 8).to(device)

    key = 'paper'
    net = xception(key, 3, 8).to(device)
    print('in:', net, key)

    in_data = torch.randint(0, 256, (24, 3, 299, 299), dtype=torch.float)
    print(in_data.shape)
    in_data = in_data.to(device)

    out_data = net(in_data)
    out_data = out_data.cpu()
    print('out:', out_data.shape)
    pass
