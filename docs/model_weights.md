# 模型权重获取与加载

权重不包含在 GitHub 仓库中。当前没有公开下载地址，请向仓库维护者获取，或按 README 的命令在真实数据上重新训练。用于预测的权重应与网络结构、输入通道和物理尺度相匹配。

## 当前 GUI 默认权重

将原双任务 U-Net 的 `best.pt` 放在项目根目录下的：

```text
experiments/reconstruction/sem_real_256_balanced/best.pt
```

它使用两通道的 256 × 256 模板输入，标定 0.78125 nm/px，包含灰度生成分支与两孔／三孔分类头。本轮基线选择第 5 轮的检查点。运行时也可以在 GUI 中选择下载到其他目录的兼容权重。

GUI 优先使用 `experiments/deployment/best.pt`。只有执行组合导出后才会生成该文件；本轮新分类器未获得可靠泛化提升，默认继续使用原 U-Net。无需人为创建这个文件。

## 独立分类器权重

`experiments/classification/<实验组>/<运行名>/best.pt` 是独立孔数分类器，只能输出分类，不能单独生成 SEM 灰度图。不要将独立分类权重直接当作 GUI 所需的完整灰度预测权重。

可通过以下命令复测独立分类器：

```powershell
python -m dsa.training.classification --checkpoint experiments/classification/resampled_v3/resnet18_balanced_sampling/best.pt --pairs data/sem_pairs_256 --split test --out outputs/checks/resnet_test
```

若后续得到合适的分类器，可按照 README 的 `scripts/deploy_classifier.py` 接口与灰度生成器组合导出。

## 权重版本记录

共享时应保留对应的网络名称、配置文件、最佳轮次、输入标定、数据划分版本和评估结果。聚合指标见 [experiment_results.md](experiment_results.md)，网络逐层定义见 [model_architectures.md](model_architectures.md)。完整训练日志留在仓库外的实验目录。

权重和数据可以分别下载，代码更新不会自动替换本地权重。`experiments/` 与常见权重扩展名均由 `.gitignore` 排除。
