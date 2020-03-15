import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=True):
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
                               padding=padding, dilation=dilation,
                               groups=in_channels, bias=bias)
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
第2-第4个module结构相似，都是stride=2的SeprableConv下采样，用_ConvBlock实现。

对MiddleFlow：
第5-第12个module结构相似，都不进行下采样，用_ConvBlock实现。

对ExitFlow：
第14个module有stride=2的SeprableConv下采样，通道数不同，用_ConvExitBlock实现。
第15个module拆开，3个卷积单独实现
"""


class XceptionBackbone(nn.Module):
    """
    实现了output stride = 16的backbone
    """

    def __init__(self, in_channels=3):
        super(XceptionBackbone, self).__init__()
        # 以下Entry Flow
        conv1 = [nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
                 nn.BatchNorm2d(32),
                 nn.ReLU(inplace=True), ]
        self.entry_conv1 = nn.Sequential(*conv1)  # 2x

        conv2 = [nn.Conv2d(32, 64, 3, padding=1, bias=False),
                 nn.BatchNorm2d(64),
                 nn.ReLU(inplace=True), ]
        self.entry_conv2 = nn.Sequential(*conv2)

        self.entry_block1 = _ConvBlock(64, 128, stride=2)  # 4x
        self.entry_block2 = _ConvBlock(128, 256, stride=2)  # 8x
        self.entry_block3 = _ConvBlock(256, 728, stride=2)  # 16x

        # 以下Middle Flow
        # 论文第5页DeepLabv3 as encoder: 16x以后使用dilation=2，不再下采样
        # 改进之一，middle block有16个
        self.middle_flow = nn.ModuleList(
            [_ConvBlock(728, 728, dilation=2)] * 16)  # 16x

        # 以下Exit Flow
        # 论文第5页DeepLabv3 as encoder: 16x以后使用dilation=2，不再下采样
        self.exit_block = _ConvExitBlock(728, 1024, stride=1, dilation=2)

        conv1 = [SeparableConv2d(1024, 1536, 3, padding=2, dilation=2, bias=False),
                 nn.BatchNorm2d(1536),
                 nn.ReLU(inplace=True), ]
        self.exit_conv1 = nn.Sequential(*conv1)

        conv2 = [SeparableConv2d(1536, 1536, 3, padding=2, dilation=2, bias=False),
                 nn.BatchNorm2d(1536),
                 nn.ReLU(inplace=True), ]
        self.exit_conv2 = nn.Sequential(*conv2)

        conv3 = [SeparableConv2d(1536, 2048, 3, padding=2, dilation=2, bias=False),
                 nn.BatchNorm2d(2048),
                 nn.ReLU(inplace=True), ]
        self.exit_conv3 = nn.Sequential(*conv3)
        self._init_param()
        pass

    def _init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        pass

    def forward(self, x):
        # Entry Flow

        x = self.entry_conv1(x)  # 2x
        x = self.entry_conv2(x)  # 2x

        x = self.entry_block1(x)  # 4x
        x = self.entry_block2(x)  # 8x
        x = self.entry_block3(x)  # 16x

        # Middle Flow
        for block in self.middle_flow:
            x = block(x)  # 16x

        # Exit Flow
        x = self.exit_block(x)  # 16x
        x = self.exit_conv1(x)  # 16x
        x = self.exit_conv2(x)  # 16x
        x = self.exit_conv3(x)  # 16x

        return x

    pass


class _ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(_ConvBlock, self).__init__()
        convs = [SeparableConv2d(inplanes, planes, 3, stride=1, padding=dilation,
                                 dilation=dilation, bias=False),  # 第1个SeparableConv2d,stride=1不下采样
                 nn.BatchNorm2d(planes),
                 nn.ReLU(inplace=True),

                 SeparableConv2d(planes, planes, 3, stride=1, padding=dilation,
                                 dilation=dilation, bias=False),  # 第2个SeparableConv2d,stride=1不下采样
                 nn.BatchNorm2d(planes),
                 nn.ReLU(inplace=True),

                 SeparableConv2d(planes, planes, 3, stride=stride,
                                 padding=dilation, dilation=dilation,
                                 bias=False),  # 第2个SeparableConv2d,可能stride=2下采样2倍
                 nn.BatchNorm2d(planes), ]
        self.convs = nn.Sequential(*convs)

        self.project = None
        if inplanes != planes or stride != 1:
            self.project = ResidualConnection(inplanes, planes, stride)
            pass
        pass

    def forward(self, x):
        identity = x  # residual connection 准备
        x = self.convs(x)
        if self.project is not None:
            identity = self.project(identity)
        x = x + identity  # residual connection 相加
        return F.relu(x, inplace=True)

    pass


class _ConvExitBlock(nn.Module):
    def __init__(self, in_channels=728, out_channels=1024, stride=1, dilation=1):
        """
        Exit Flow的第1个module
        前两个Separable Conv都不做下采样
        最后一个Separable Conv下采样2倍
        :param in_channels: 输入channels
        :param out_channels: 输出channels
        """
        super(_ConvExitBlock, self).__init__()
        convs = [SeparableConv2d(in_channels, in_channels, 3, stride=1,
                                 padding=dilation, dilation=dilation,
                                 bias=False),  # 728->728，不下采样
                 nn.BatchNorm2d(in_channels),
                 nn.ReLU(inplace=True),
                 SeparableConv2d(in_channels, out_channels, 3, stride=1,
                                 padding=dilation, dilation=dilation,
                                 bias=False),  # 728->1024，不下采样
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(inplace=True),
                 SeparableConv2d(out_channels, out_channels, 3, stride=stride,
                                 padding=dilation, dilation=dilation, bias=False),  # 1024->1024，下采样2倍
                 nn.BatchNorm2d(out_channels), ]
        self.convs = nn.Sequential(*convs)

        self.project = None
        if in_channels != out_channels or stride != 1:
            self.project = ResidualConnection(in_channels, out_channels, stride)
            pass
        pass

    def forward(self, x):
        identity = x  # residual connection 准备
        x = self.convs(x)  # 下采样2倍
        if self.project is not None:
            identity = self.project(identity)
        x = x + identity  # residual connection 相加
        return F.relu(x, inplace=True)

    pass


class _XceptionBackBoneFactory(nn.Module):
    def __init__(self, in_channels, entry_planes=(128, 256, 728),
                 output_stride=16, n_middle=16):
        """
        实现一个工厂类，将DeepLabV+论文改进的XceptionBackbone统一到一起。
        :param inplanes: 输入channels，图像channels
        :param entry_planes: 每个entry block的输出channels，列表类型
        :param n_middle: 改进实现用16
        :param output_stride: 输入和主干输出的比值，默认16x
        """
        super(_XceptionBackBoneFactory, self).__init__()
        assert len(entry_planes) == 3  # Entry Block共3个
        """
        DeepLabV3论文第3页原文:
        If one would like to double the spatial density of computed feature
        responses in the DCNNs (i.e., output stride = 16), the
        stride of last pooling or convolutional layer that decreases
        resolution is set to 1 to avoid signal decimation. Then, all
        subsequent convolutional layers are replaced with atrous
        convolutional layers having rate r = 2.
        下采样采用stride=2，dilation=1的卷积。
        下采样以后stride=1不再下采样，dilation根据output stride不同取相应的值
        ------------------------------------------------------------------------
        DeepLabV3+论文第5页原文:
        Here, we denote output stride as the ratio of input image spatial resolution 
        to the final output resolution (before global pooling or fully-connected layer). 

        For the task of image classification, the spatial resolution of the
        final feature maps is usually 32 times smaller than the input image resolution and
        thus output stride = 32. 

        For the task of semantic segmentation, one can adopt
        output stride = 16 (or 8) for denser feature extraction by removing the striding
        in the last one (or two) block(s) and applying the atrous convolution correspond-
        ingly (e.g., we apply rate = 2 and rate = 4 to the last two blocks respectively
        for output stride = 8).
        ------------------------------------------------------------------------
        如果output stride=8，那么在8x之后，所有卷积stride=1，dilation=4
        如果output stride=16，那么在16x之后，所有卷积stride=1,dilation=2
        如果output stride=32，那么在32x之后，所有卷积stride=1，dilation=1
        
        strides
        对于8x，最后下采样是上一个block进行的，后面的dilation[0]=4,dilation[1]=4
        对于16x，最后下采样是在stride[0]进行的，dilation[0]=1;后面dilation[1]=2
        对于32x，最后下采样是在stride[1]进行的，dilation[0]和dilation[1]都是1
        """
        if output_stride == 8:
            strides = (1, 1)
            dilations = (4, 4)
        elif output_stride == 16:
            strides = (2, 1)
            dilations = (1, 2)
        elif output_stride == 32:
            strides = (2, 2)
            dilations = (1, 1)
        else:
            raise ValueError('output stride error!')

        # 以下Entry Flow
        conv1 = [nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
                 nn.BatchNorm2d(32),
                 nn.ReLU(inplace=True), ]
        self.entry_conv1 = nn.Sequential(*conv1)  # 第1个普通卷积，2x

        conv2 = [nn.Conv2d(32, 64, 3, padding=1, bias=False),
                 nn.BatchNorm2d(64),
                 nn.ReLU(inplace=True), ]
        self.entry_conv2 = nn.Sequential(*conv2)  # 第2个普通卷积,2x

        self.entry_block1 = _ConvBlock(64, entry_planes[0], stride=2)  # 4x
        self.entry_block2 = _ConvBlock(entry_planes[0], entry_planes[1], stride=2)  # 8x
        self.entry_block3 = _ConvBlock(entry_planes[1], entry_planes[2],
                                       stride=strides[0], dilation=dilations[0])
        # stride=1是8x,dilation应为4
        # stride=2是16x，dilation应为1

        # 以下Middle Flow
        inplanes = entry_planes[2]
        mid_block = _ConvBlock(inplanes, inplanes,
                               stride=1, dilation=dilations[1])  # 不再下采样，根据output stride情况决定dilation
        self.middle_blocks = nn.ModuleList([mid_block] * n_middle)  # 重复n_middle次

        # 以下Exit Flow
        self.exit_block = _ConvExitBlock(inplanes, 1024, stride=strides[1],  # 和前面stride[0]结合使用，决定8x、16x、32x
                                         dilation=dilations[1])

        conv1 = [SeparableConv2d(1024, 1536, 3, padding=dilations[1],
                                 dilation=dilations[1], bias=False),
                 nn.BatchNorm2d(1536),
                 nn.ReLU(inplace=True), ]
        self.exit_conv1 = nn.Sequential(*conv1)

        conv2 = [SeparableConv2d(1536, 1536, 3, padding=dilations[1],
                                 dilation=dilations[1], bias=False),
                 nn.BatchNorm2d(1536),
                 nn.ReLU(inplace=True), ]
        self.exit_conv2 = nn.Sequential(*conv2)

        conv3 = [SeparableConv2d(1536, 2048, 3, padding=dilations[1],
                                 dilation=dilations[1], bias=False),
                 nn.BatchNorm2d(2048),
                 nn.ReLU(inplace=True), ]
        self.exit_conv3 = nn.Sequential(*conv3)

        self._init_param()
        pass

    def _init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        pass

    def forward(self, x):
        # Entry Flow
        x = self.entry_conv1(x)  # 2x
        x = self.entry_conv2(x)  # 2x
        x = self.entry_block1(x)  # 4x
        low_level_features = x
        x = self.entry_block2(x)
        x = self.entry_block3(x)

        # Middle Flow
        for block in self.middle_blocks:
            x = block(x)
            pass

        # Exit Flow
        x = self.exit_block(x)
        x = self.exit_conv1(x)
        x = self.exit_conv2(x)
        x = self.exit_conv3(x)
        return x, low_level_features

    pass


def xception_backbone(in_channels, output_stride=16):
    if output_stride == 8:
        return _XceptionBackBoneFactory(in_channels, output_stride=output_stride)
    elif output_stride == 16:
        return _XceptionBackBoneFactory(in_channels, output_stride=output_stride)
    elif output_stride == 32:
        return _XceptionBackBoneFactory(in_channels, output_stride=output_stride)
    else:
        raise ValueError('output stride error!')


################################################################################

if __name__ == '__main__':
    # device = torch.device('cuda:6')
    device = torch.device('cpu')
    # net = XceptionBackbone(3).to(device)
    net = xception_backbone(3, output_stride=8).to(device)
    print('in:', net)

    in_data = torch.randint(0, 256, (24, 3, 299, 299), dtype=torch.float)
    print(in_data.shape)
    in_data = in_data.to(device)

    high_level, low_level = net(in_data)
    high_level = high_level.cpu()
    low_level = low_level.cpu()
    print('out:', high_level.shape, low_level.shape)
    pass
