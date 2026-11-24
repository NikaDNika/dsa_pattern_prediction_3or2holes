# DSA 模板形貌与孔图案预测

当前项目以模板轮廓和侧壁距离图为输入，预测 SEM 灰度形貌及两孔／三孔。GUI 依次显示原图、二值模板和黄色预测区域，底部显示孔数分类及分类分数。

## 快速启动

```powershell
git clone https://github.com/NikaDNika/dsa_pattern_prediction_3or2holes.git
cd dsa_pattern_prediction_3or2holes
```

仓库只包含代码、配置、测试及文档。真实数据、模型权重和运行输出单独保存，不随 GitHub 克隆下载。GUI 预测需要先获取灰度模型权重；真实数据训练需要恢复数据目录。具体格式与放置位置见 [数据说明](docs/dataset.md) 和 [模型权重说明](docs/model_weights.md)。当前没有公开的数据或权重下载地址，请向仓库维护者获取。

在项目根目录安装依赖后双击 `launch_gui.cmd`，或执行：

```powershell
python -m pip install -r requirements.txt
pwsh -NoProfile -STA -File apps/predict_gui.ps1
```

“填入演示样例”使用真实数据留出点位的 SEM 裁片。GUI 优先使用 `experiments/deployment/best.pt`；该文件尚未导出时，使用 `experiments/reconstruction/sem_real_256_balanced/best.pt`。

统一后的单元图覆盖 **200 nm × 200 nm**，分辨率 **256 × 256**，标定填 **0.78125 nm/px**，ROI 留空。整幅阵列图需填写 ROI 的“左 上 宽 高”，单位为输入像素，从零起算。原始图填其实际标定。已提取的白腔体二值模板选择“二值模板（白腔体）”。

预测输出写入 `outputs/gui/gui-日期时间-编号/`。GPU 不可用时程序可以使用 CPU。若 Python 环境需要指定，启动参数可加 `-Python "C:\path\to\python.exe"`。

## 文件结构

```text
dsa_pattern_prediction/
├── launch_gui.cmd                   Windows 启动入口
├── README.md
├── requirements.txt
├── apps/
│   └── predict_gui.ps1              拖放、参数填写与三栏结果展示
├── configs/
│   ├── classification_comparison.json    模型和学习率对照实验
│   ├── classification_balanced.json      类别权重对照实验
│   └── classification_resampled.json     平衡重采样与较长训练实验
├── docs/
│   ├── model_architectures.md       各网络逐层结构、空间尺寸和参数量
│   ├── dataset.md                   仓库外数据获取、格式与目录恢复
│   ├── model_weights.md             仓库外权重与 GUI 加载说明
│   └── experiment_results.md        可随代码共享的汇总指标
├── dsa/
│   ├── __main__.py                  分割与图像推理命令行
│   ├── data.py                     掩膜校验、物理网格和 SDF
│   ├── engine.py                   图像推理、叠加和旧掩膜训练流程
│   ├── metrics.py                  连通区与分类指标
│   ├── splitting.py                分组划分
│   ├── synthetic.py                合成演示数据
│   ├── models/
│   │   ├── unet.py                 共享编码器、灰度解码器和分类头
│   │   └── classifiers.py          U-Net 编码器、ResNet-18、ConvNeXt-Tiny
│   ├── preprocessing/
│   │   ├── segmentation.py         原始模板图二值化
│   │   └── pairs.py                提取模板、配对标签、物理点位划分
│   └── training/
│       ├── common.py               共用配对数据读取、批处理和混合精度
│       ├── reconstruction.py       SEM 重建训练、评估及暗区选取
│       ├── classification.py       配置驱动的分类训练和模型对照
│       └── pore_shapes.py          历史独立孔形实验
├── scripts/
│   ├── deploy_classifier.py         合并已选分类器与灰度生成器
│   └── report_experiments.py        汇总各组完整指标与结构快照
├── examples/                        清单格式示例，本地测试照片不上传
├── data/
│   ├── 2D RE R25 RH32 提取-WZY-20261122/  原始 TIFF
│   ├── 2D_RE_R25_RH32_统一200nm_256/      本次训练的灰度标签
│   ├── 2D_RE_R25_RH32_统一200nm_256_灰底/ 保留的另一版裁片
│   ├── sem_pairs_256/               模板 PNG、pairs.csv 和像素缓存
│   ├── demo/                        合成样本
│   └── sem_r2_c07/                  屏摄提取的历史演示样例
├── experiments/
│   ├── reconstruction/             原灰度模型及评估
│   ├── classification/             分类对照，每次运行独立子目录
│   ├── deployment/                 导出后生成，供 GUI 使用的组合权重
│   └── archive/                    历史模型权重和旧实验记录
├── outputs/
│   ├── gui/                        GUI 预测记录
│   └── checks/                     问题复现与命令行核对记录
└── tests/                          尺度、分组、分割和叠加回归测试
```

整理时将原 `runs/` 中模型、GUI 记录和诊断分开迁移。删除了可重新生成的合成预测图、临时分割样例和独立孔形实验的标签缓存。原始 TIFF、单元裁片、训练配对、现用模型、历史模型权重、用户测试图和 GUI 记录保留。旧代码模块名已迁移，下面的命令使用新位置。数据清单保留原标签路径，训练数据本身没有迁移。

上面的 `data/`、`experiments/`、`outputs/` 描述恢复数据或运行后形成的本地目录，它们均被 `.gitignore` 排除，不存在于代码仓库中。添加 Git 忽略规则不会删除本地文件。

## 数据配对与划分

输入包含两个通道，第一通道是填实内部后的二值模板，第二通道是有符号距离图，内部为正，外部为负，按 24 nm 截断。两通道均由模板生成，最终孔灰度只作为训练目标。

`data/sem_pairs_256/pairs.csv` 记录单元 ID、来源、模板和原灰度标签、点位、划分与标定。`paired_pixels.npy` 是按索引对应的 8 位模板／距离图／灰度标签缓存。

| 集合 | 两孔 | 三孔 | 孔数待复核 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 训练 | 251 | 9067 | 289 | 9607 |
| 验证 | 52 | 1682 | 64 | 1798 |
| 测试 | 53 | 2133 | 68 | 2254 |

同一物理点位在不同倍率、分辨率和拍摄序号下的图像属于同一集合。16 个点位分为训练 10 个、验证 3 个、测试 3 个。边缘截断的 585 对保留在配对目录，排除于模型训练和评估。孔数待复核的完整样本仅用于灰度重建，分类实验只使用两孔／三孔标签。

本阶段轮廓来自 DSA 后图。真实 DSA 前数据上的效果需要同单元的前后配对图检验。孔数标签由图像处理初分，正式比较模型前仍应人工核验少数类标签。多倍率重复拍摄可能提高某些单元的样本权重。

重新生成配对时使用新输出目录：

```powershell
python -m dsa.preprocessing.pairs --raw 'data/2D RE R25 RH32 提取-WZY-20261122' --images 'data/2D_RE_R25_RH32_统一200nm_256' --out data/sem_pairs_256_new
```

## 模型的各层结构

| 网络 | 阶段和通道 | 分类输出 |
| --- | --- | --- |
| 现用双任务 U-Net | 256²×2 输入；每块两层 3×3 卷积、GroupNorm、SiLU；编码通道 8/16/32，三次 2×2 池化；瓶颈 32²×64 | 自适应池化到 2×2，展平 256，Linear 256→32，SiLU，Dropout 0.15，Linear 32→2 |
| 单任务 U-Net 编码器 | 同样的三阶段编码器、瓶颈和分类头，单独优化分类 | 两孔／三孔 |
| ResNet-18 | 2 通道 7×7 stride 2 stem；3×3 最大池化；四阶段 BasicBlock 数 [2,2,2,2]，通道 [64,128,256,512]；每块两层 3×3 卷积及残差连接 | 全局平均池化，Linear 512→2 |
| ConvNeXt-Tiny | 2 通道 4×4 stride 4 stem；四阶段块数 [3,3,9,3]，通道 [96,192,384,768]；块内 7×7 深度卷积、LayerNorm、通道扩展 4 倍的两层线性变换、GELU 和残差连接 | 全局平均池化，LayerNorm，Linear 768→2 |

U-Net 的灰度解码器使用双线性上采样与编码器特征拼接，通道依次 32/16/8，最后 1×1 卷积及 Sigmoid 生成 256×256 灰度图。独立分类器可替换孔数分类部分，灰度生成器继续输出黄色暗区。

ResNet 与 ConvNeXt 使用 torchvision 官方实现，将首层改为两通道、末层改为两类。当前对照从随机初始化训练。结构和实现来源见 [ResNet](https://docs.pytorch.org/vision/stable/models/resnet.html) 与 [ConvNeXt](https://docs.pytorch.org/vision/stable/models/convnext.html)。

逐层运算、输出尺寸、归一化与激活函数见 [模型结构说明](docs/model_architectures.md)。现用 U-Net 为 130,683 参数；ResNet-18 为 11,174,402 参数；ConvNeXt-Tiny 为 27,820,130 参数。

## 训练超参数

原灰度重建实验的设定如下。

| 参数 | 当前基线 |
| --- | --- |
| 优化器 | AdamW，weight decay 0.01 |
| 初始学习率 | 0.001 |
| 调度 | CosineAnnealing，每轮更新，末端学习率 0.0001 |
| 批量 | 24 |
| 训练轮数 | 6，按验证集选择检查点 |
| 网络宽度 | 8，约 13 万参数 |
| 随机种子 | 42 |
| 两孔重采样 | 每轮约 18% |
| 分类权重 | 按训练频数倒数平方根归一化 |
| 损失 | 0.85×模板内 MAE + 0.15×背景 MAE + 0.10×腔体灰度梯度误差 + 0.12×分类交叉熵 |
| 检查点选择 | 验证集模板内 MAE + 0.15×(1−宏 F1)，越小越好 |

分类对照由 JSON 配置控制学习率、批量、轮数、早停耐心值、weight decay、标签平滑、类别权重幂和翻转增强。所有模型保持 256×256 输入，使用同一物理点位划分及未重采样的验证／测试数据。权重幂 0.5 为倒数平方根、1.0 为逆频率加权、0 为不加权。新分类实验仅优化分类交叉熵。

| 实验配置 | 学习率 | batch | 最大轮数／早停耐心值 | 类别处理 | weight decay |
| --- | --- | --- | --- | --- | --- |
| comparison | U-Net 1e-3；ResNet 3e-4、1e-4；ConvNeXt 1e-4 | 24；ConvNeXt 12 | 8／4 | 自然分布，逆频率平方根权重 | 0.01 |
| balanced | ResNet、ConvNeXt 1e-4 | 24、12 | 10／5 | 自然分布，逆频率权重 | 0.01、0.05 |
| resampled | ResNet 3e-4；ConvNeXt 1e-4 | 24、12 | 20／7，至少训练 10 轮 | 每轮两孔和三孔各采样 50%，有放回，交叉熵不加权 | 0.01、0.05 |

新实验统一使用 AdamW、余弦学习率调度（最低为初始学习率的 5%）、梯度范数裁剪 5、随机种子 42、标签平滑 0。每批独立以 50% 概率水平／垂直翻转，验证和测试关闭增强。GPU 使用 bfloat16 混合精度；输入仅含模板形状。重采样实验每轮样本总量不变，验证和测试保留原类别比例。

本次环境为 PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、RTX 4060 Laptop 8 GB。报告脚本在 `experiments/classification/runtime.json` 记录运行环境。

```powershell
python -m dsa.training.classification --config configs/classification_comparison.json
python -m dsa.training.classification --config configs/classification_balanced.json
python -m dsa.training.classification --config configs/classification_resampled.json
python scripts/report_experiments.py
```

重复实验应在配置中更换 `out`，新目录名会避免覆盖历史实验。每个模型保存 `config.json`、`architecture.txt`、`history.json`、`best.pt`、`metrics.json` 和训练／测试分数。综合结果为 `comparison.json`，`selected.json` 只依据验证集宏 F1 选出模型。两孔精确率、召回率、均衡准确率和混淆矩阵共同用于解释总体准确率。

完整结果汇总为 `experiments/classification/comparison.md` 和同名 CSV。测试集标签用于各组末尾报告；后续实验参数根据训练和验证表现调整。已经反复查看过的测试集应视为开发阶段的留出集，最终对外报告还需新增独立点位测试。

无需下载实验目录即可查看 [已完成实验的汇总指标](docs/experiment_results.md)。该文档为本轮实验的静态结果记录。

本轮完成 8 组新分类实验。以下两行深层模型均采用 50% 两孔重采样、最多 20 轮，并由验证集选择检查点；它们用于比较分类能力，灰度生成器仍为 U-Net。

| 模型／训练方案 | 训练准确率 | 留出集准确率 | 留出集两孔召回率 | 留出集宏 F1 |
| --- | ---: | ---: | ---: | ---: |
| 原双任务 U-Net | 65.44% | 89.98% | 26.42% | 0.530 |
| ResNet-18，平衡采样 | 98.36% | 95.24% | 3.77% | 0.506 |
| ConvNeXt-Tiny，平衡采样 | 85.99% | 87.19% | 18.87% | 0.499 |

自然采样的几组模型几乎全判三孔，虽然留出集准确率达到 97.58%，两孔召回率为 0。原模型同样存在两孔误报多的问题；本次实测尚未得到可替换默认模型的可靠提升。GUI 继续使用原权重，未生成 `experiments/deployment/best.pt`。

ResNet 的训练拟合明显增强，留出点位表现却较弱，提示过拟合或点位分布差异。下一阶段优先增加独立点位、人工核验两孔标注，并尝试预训练微调与正则化；只增加深度或轮数不能保证改善。真实 DSA 前后同单元配对将进一步检验模板形貌对孔数的可预测程度。

复测已训练的独立分类模型无需重新训练，可将 `--split` 改为 `train`、`val` 或 `test`：

```powershell
python -m dsa.training.classification --checkpoint experiments/classification/comparison_v1/resnet18_lr3e4/best.pt --pairs data/sem_pairs_256 --split test --out outputs/checks/resnet_test
```

灰度模型重新训练：

```powershell
python -m dsa.training.reconstruction train --pairs data/sem_pairs_256 --out experiments/reconstruction/new_run --epochs 6 --batch 24 --width 8 --two-hole-fraction 0.18 --device cuda
```

## 推理与 GUI

原图一键分割与推理：

```powershell
python -m dsa predict-image --image 'data/2D_RE_R25_RH32_统一200nm_256/2孔/F015_R04_C02.png' --nm-per-px 0.78125 --checkpoint experiments/reconstruction/sem_real_256_balanced/best.pt --out outputs/checks/new_sample --dark-only-overlay
```

二值模板使用 `--binary-template`。原图分割可用 `--polarity dark`、`--threshold 0.45` 或 `--sigma 0` 调节。ROI 用 `--roi 左 上 宽 高`。

已提取的标准模板可直接执行灰度预测：

```powershell
python -m dsa.training.reconstruction predict --template data/sem_pairs_256/templates/F048_R02_C18.png --checkpoint experiments/reconstruction/sem_real_256_balanced/best.pt --out outputs/checks/new_template
```

验证集优胜分类器可与灰度生成器导出为组合权重。以下给出导出接口；本轮结果建议继续使用原权重。导出要求分类器验证宏 F1 优于原模型，且能识别验证集中的两类：

```powershell
python scripts/deploy_classifier.py --comparison experiments/classification/balanced_loss_v2 --sem experiments/reconstruction/sem_real_256_balanced/best.pt --out experiments/deployment
```

## 灰度、暗度和文件输出

灰度预测经 Sigmoid 取值 0～1，再乘 255 取整，得到 0～255 的灰度级。0 最暗、255 最亮。暗度等于腔体内部灰度中位数减去候选中心半径 3 nm 内的平均灰度，数值越大表示相对更暗。它衡量图像对比度，物理孔深需另外测量。

程序按分类孔数，从独立暗核心中按暗度选择候选。候选不足时在 JSON 内记录。GUI 只显示黄色层；命令行不加 `--dark-only-overlay` 时可生成带红色候选边界的诊断叠加。

| 输出 | 用途 |
| --- | --- |
| input_preview.png | 与原图同坐标的输入预览 |
| segmentation/template.png | 二值模板 |
| segmentation/overlay.png | 分割边界核对 |
| prediction/predicted_sem.png | 256×256 灰度预测 |
| predicted_dark_regions_on_input.png | 原图坐标下的黄色区域掩膜 |
| predicted_holes_on_input.png | 按分类数目选出的候选掩膜 |
| overlay_on_input.png | GUI 的黄色叠加图 |
| prediction/prediction.json | 分类分数、候选排名、暗度及复核字段 |

## 检查与历史流程

```powershell
python -m unittest discover -s tests -v
```

`python -m dsa --help` 可查看原二值孔掩膜模型的训练、评估、批量预测命令。示例清单位于 `examples/manifest.example.csv`。其四类标签为 `good`、`line_or_bridge`、`wrong_count`、`other_defect`。这些命令需要配准的二值孔标签。旧演示模型位于 `experiments/archive/demo-train-v2/best.pt`。
