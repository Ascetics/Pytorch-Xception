# Pytorch-Xception
复现Xception论文

更详细的解读参看博客https://blog.csdn.net/qpeity/article/details/104663894

# Xception结构
Xception脱胎于Inception，Inception的思想是将卷积分成cross-channel conv和spatial conv。
Xception本质上是将cross-channel conv和spatial conv完全解耦。

Xception的特征提取基础由36个conv layer构成。
这36个conv layer被组织成14个module，除了第一个和最后一个module，其余的module都带有residual connection（残差，参看何凯明大神的ResNet）。
简言之，Xception结构就是连续使用depthwise separable convolution layer和residual connection。

14个module分成三个部分Entry flow、Middle flow、Exit flow，最后根据实际需要加入FC。

Entry flow的output stride=16x

Middle flow的output stride=16x

Exit flow的output stride，在Avgpool之前是32x，在Avg Pool之后是个2048 dim的向量。

# 实现细节

输入先经过Entry flow，不重复；再经过Middle flow，Middle flow重复8次；最后经过Exit flow，不重复。

所有的Conv 和 Separable Conv后面都加BN层，但是论文Figure 5没有画出来。

所有的Separable Conv都用depth=1，也就是每个depth-wise都是“切片”的。

注意， depthwise separable convolution在spatial conv和cross-channel conv之间不要加ReLU激活函数，任何激活函数都不要加。论文Figure 10展示了，这里不加激活函数效果最好，加ReLU、ELU都不好。
