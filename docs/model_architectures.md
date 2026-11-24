# 网络逐层说明

张量以 N×C×H×W 表示。N 为批量，两路输入分别为二值模板与有符号距离图，空间标定均为 0.78125 nm/px。

## 现用双任务 U-Net

卷积块 B(a,b) 为 Conv2d(a,b,3,padding=1,bias=False) → GroupNorm(4,b) → SiLU → Conv2d(b,b,3,padding=1,bias=False) → GroupNorm(4,b) → SiLU。

| 顺序 | 运算 | 输出 C×H×W |
| --- | --- | --- |
| 输入 | 模板与距离图 | 2×256×256 |
| 编码 1 | B(2,8)，保留跳跃特征 | 8×256×256 |
| 下采样 1 | MaxPool 2×2 | 8×128×128 |
| 编码 2 | B(8,16)，保留跳跃特征 | 16×128×128 |
| 下采样 2 | MaxPool 2×2 | 16×64×64 |
| 编码 3 | B(16,32)，保留跳跃特征 | 32×64×64 |
| 下采样 3 | MaxPool 2×2 | 32×32×32 |
| 瓶颈 | B(32,64) | 64×32×32 |
| 分类分支 | AdaptiveAvgPool(2,2) → Flatten → Linear(256,32) → SiLU → Dropout(0.15) → Linear(32,2) | 2 个 logits |
| 解码 1 | 双线性放大到 64²，拼接编码 3，B(96,32) | 32×64×64 |
| 解码 2 | 双线性放大到 128²，拼接编码 2，B(48,16) | 16×128×128 |
| 解码 3 | 双线性放大到 256²，拼接编码 1，B(24,8) | 8×256×256 |
| 灰度输出 | Conv2d(8,1,1) → Sigmoid | 1×256×256 |

共 130,683 个参数。分类器对照中的 `unet_encoder` 保留编码、瓶颈与分类分支，去掉解码分支，共 82,066 个参数。

## ResNet-18

每个 BasicBlock 主支为 3×3 卷积 → BatchNorm → ReLU → 3×3 卷积 → BatchNorm；与捷径相加后 ReLU。需要缩小尺寸或改变通道时，捷径使用 1×1 卷积与 BatchNorm。每阶段首块的步长决定是否下采样。

| 顺序 | 运算 | 输出 C×H×W |
| --- | --- | --- |
| stem | Conv2d(2,64,7,stride=2,padding=3) → BatchNorm → ReLU | 64×128×128 |
| 池化 | MaxPool(3,stride=2,padding=1) | 64×64×64 |
| stage 1 | 2 个 BasicBlock，stride 1 | 64×64×64 |
| stage 2 | 2 个 BasicBlock，首块 stride 2 | 128×32×32 |
| stage 3 | 2 个 BasicBlock，首块 stride 2 | 256×16×16 |
| stage 4 | 2 个 BasicBlock，首块 stride 2 | 512×8×8 |
| 分类头 | GlobalAvgPool → Flatten → Linear(512,2) | 2 个 logits |

本项目版本共 11,174,402 个参数。名称中的 18 按标准 ResNet 命名计算卷积主干及全连接层，归一化和激活不单独计入。

## ConvNeXt-Tiny

每个块为 7×7 深度卷积（padding 3，每通道独立）→ 转为 NHWC → LayerNorm(eps=1e-6) → Linear(C,4C) → GELU → Linear(4C,C) → 转回 NCHW → 可学习的 layer scale → stochastic depth → 与输入相加。layer scale 初始化为 1e-6；stochastic depth 按块逐步增加，最大 0.1。

| 顺序 | 运算 | 输出 C×H×W |
| --- | --- | --- |
| stem | Conv2d(2,96,4,stride=4) → LayerNorm2d | 96×64×64 |
| stage 1 | 3 个块 | 96×64×64 |
| 下采样 1 | LayerNorm2d → Conv2d(96,192,2,stride=2) | 192×32×32 |
| stage 2 | 3 个块 | 192×32×32 |
| 下采样 2 | LayerNorm2d → Conv2d(192,384,2,stride=2) | 384×16×16 |
| stage 3 | 9 个块 | 384×16×16 |
| 下采样 3 | LayerNorm2d → Conv2d(384,768,2,stride=2) | 768×8×8 |
| stage 4 | 3 个块 | 768×8×8 |
| 分类头 | GlobalAvgPool → LayerNorm2d → Flatten → Linear(768,2) | 2 个 logits |

本项目版本共 27,820,130 个参数。Tiny 是 ConvNeXt 系列的型号，不能据此理解为比现有 U-Net 更小。

ResNet 和 ConvNeXt 使用 torchvision 官方结构，首层改成 2 通道，末层改成 2 类；当前实验未使用 ImageNet 预训练权重。分类 logits 经 softmax 得到未校准的分类分数。

这两种新增网络用于替换孔数分类器。灰度图仍由 U-Net 生成；组合导出时保留原灰度生成权重。其效果应分别用分类指标与灰度误差检验。

实现：[unet.py](../dsa/models/unet.py)、[classifiers.py](../dsa/models/classifiers.py)。官方说明：[ResNet](https://docs.pytorch.org/vision/stable/models/resnet.html)、[ConvNeXt](https://docs.pytorch.org/vision/stable/models/convnext.html)。
