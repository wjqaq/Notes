⌚2016:CVPR[@heDeepResidualLearning2015]
##### 👀研究背景
- **退化问题**：随着网络层数增加，**训练集准确率先上升后下降**，测试集准确率同步下降，且并非过拟合导致（过拟合表现为训练集准确率高、测试集低）；
- **梯度消失 / 爆炸**：早期深层网络训练困难的表面原因，可通过**批归一化（BN）**、权重初始化等方法有效缓解，但 BN 无法解决退化问题 —— 这说明退化是深层网络的**本质学习问题**，而非数值计算问题。
##### 🤖模型架构
![](assets/ResNet/file-20260305140817002.png)
- （layer < 50)每层层内输入输出通道数相同，直接将这一层的当前输入给这一层的下一层输入即可：如conv2_2 input = conv2_1 input + F(conv2_1 input)，layer >= 50 层间进行 x 的降为操作（kernel_size = 1 stride = 1 padding = 0) 进行尺寸改变后再 升维操作（kernel_size = 1 stride = 1 padding = 0)这样有助于减少计算量。
- 层间输入输出通道不同，如conv2 -- conv3 需要进行 x 的升维操作并进行下采样（kernel_size = 1 stride = 2 padding = 0)
```python
import torch

from torch import nn

from models.BaseModel import BaseModel

from models.registry import ModelRegistry

  

class BasicBlock(nn.Module):

    """

    ResNet的基本残差块

    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):

        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):

        identity = x

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        if self.downsample is not None:

            identity = self.downsample(x)

        out += identity

        out = self.relu(out)

        return out

  

class Bottleneck(nn.Module):

    """

    ResNet的瓶颈残差块

    """

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):

        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):

        identity = x

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        out = self.relu(out)

        out = self.conv3(out)

        out = self.bn3(out)

        if self.downsample is not None:

            identity = self.downsample(x)

        out += identity

        out = self.relu(out)

        return out

  

@ModelRegistry.register("resnet18")

class ResNet18(BaseModel):

    """

    ResNet-18模型

    """

    def __init__(self, num_classes=10):

        super(ResNet18, self).__init__()

        self.in_channels = 64

        # 初始卷积层

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差块

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)

        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)

        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)

        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # 分类层

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):

        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:

            downsample = nn.Sequential(

                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm2d(out_channels * block.expansion)

            )

        layers = []

        layers.append(block(self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):

            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

  

@ModelRegistry.register("resnet34")

class ResNet34(BaseModel):

    """

    ResNet-34模型

    """

    def __init__(self, num_classes=10):

        super(ResNet34, self).__init__()

        self.in_channels = 64

        # 初始卷积层

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差块

        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=1)

        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)

        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)

        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)

        # 分类层

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):

        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:

            downsample = nn.Sequential(

                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm2d(out_channels * block.expansion)

            )

        layers = []

        layers.append(block(self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):

            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

  

if __name__ == '__main__':

    # 测试ResNet-18

    model = ResNet18(num_classes=10)

    print("ResNet-18 model created")

    model.summary(input_size=(1, 3, 32, 32))

    # 测试ResNet-34

    model = ResNet34(num_classes=10)

    print("\nResNet-34 model created")

    model.summary(input_size=(1, 3, 32, 32))
```
##### 💡核心方法
- **恒等捷径（Identity Shortcut）**：当输入与分支输出的特征图尺寸、通道数完全一致时，直接将输入与分支输出相加，无任何额外参数和计算量，是 ResNet 的核心设计。
- **投影捷径（Projection Shortcut）**：当特征图下采样（尺寸减半）或通道数翻倍时，通过 1×1 卷积（步长与下采样一致）对输入进行线性变换，匹配分支输出的尺寸和通道数，仅在维度不匹配时使用。
##### 🎨关键创新
- 通过残差块加深网络的同时保持模型的收敛以及避免模型退化问题；
- 通过瓶颈残差块的设计减少了更深的模型的参数量与计算量；
##### 🚀实验结果
![](assets/ResNet/file-20260305143609297.png)
模型越深效果越好、考虑计算成本问题，投影捷径只在输入输出通道数不同时采用；
##### 📈影响
- 打破了深度模型模型退化问题；
- 成为CV领域的骨干网络；
- 其思想渗透NLP领域。