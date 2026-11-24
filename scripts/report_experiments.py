"""Summarize the finished classifier experiments without changing GUI model selection."""
import csv
import json
from pathlib import Path
import sys
import platform

root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from dsa.models.classifiers import build_classifier
import torch
import torchvision
torch.set_num_threads(2)
output = root/"experiments/classification"
runtime=dict(python=platform.python_version(),torch=str(torch.__version__),
             torchvision=str(torchvision.__version__),cuda=torch.version.cuda,
             gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
(output/"runtime.json").write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
rows = []
settings = []
base = root/"experiments/reconstruction/sem_real_256_balanced"
train = json.loads((base/"train_split_metrics.json").read_text(encoding="utf-8"))
test = json.loads((base/"test/metrics.json").read_text(encoding="utf-8"))
state = torch.load(base/"best.pt",map_location="cpu",weights_only=True)
baseline = dict(
    experiment="unet_joint_original",model="unet_joint",epoch=state["epoch"],
    train_accuracy=train["accuracy"], validation_macro_f1=state["val"]["macro_f1"],
    test_accuracy=sum(test["confusion_matrix"][i][i] for i in (0,1))/sum(test["class_support"]),
    test_macro_f1=test["macro_f1"],test_two_recall=test["two_hole_recall"],
    test_two_precision=test["confusion_matrix"][0][0]/max(1,sum(r[0] for r in test["confusion_matrix"])),
    test_three_recall=test["three_hole_recall"])
rows.append(baseline)
for summary in sorted(output.glob("*/comparison.json")):
    data=json.loads(summary.read_text(encoding="utf-8"))
    for record in data:
        cfg=record["config"]
        architecture=summary.parent/cfg["name"]/"architecture.txt"
        if not architecture.exists():
            architecture.write_text(str(build_classifier(cfg["model"]))+"\n",encoding="utf-8")
        trained=len(json.loads((summary.parent/cfg["name"]/"history.json").read_text(encoding="utf-8")))
        settings.append(dict(name=cfg["name"],lr=cfg["lr"],batch=cfg["batch"],wd=cfg["weight_decay"],
                             fraction=cfg.get("two_hole_fraction"),weight=cfg["weight_power"],epochs=trained))
        rows.append(dict(experiment=cfg["name"],model=cfg["model"],epoch=record["best_epoch"],
            train_accuracy=record["train"]["accuracy"],validation_macro_f1=record["validation"]["macro_f1"],
            test_accuracy=record["test"]["accuracy"],test_macro_f1=record["test"]["macro_f1"],
            test_two_recall=record["test"]["two_recall"],test_two_precision=record["test"]["two_precision"],
            test_three_recall=record["test"]["three_recall"]))
with (output/"comparison.csv").open("w",encoding="utf-8-sig",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
text=["# 分类模型对照结果","",
      "所有新分类模型从随机初始化训练，使用相同 256×256 模板与距离图。每组按验证宏 F1 选择检查点，测试标签只用于末尾报告。",
      "原模型行使用此前已保存的评估，GPU 数值精度和批量内核变化可能使阈值附近少量预测发生变化。",
      "","| 实验 | 最佳轮 | 训练准确率 | 验证宏 F1 | 测试准确率 | 测试两孔召回 | 测试两孔精确率 | 测试宏 F1 |",
      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for r in rows:
    text.append(f"| {r['experiment']} | {r['epoch']} | {r['train_accuracy']:.2%} | {r['validation_macro_f1']:.3f} | "
                f"{r['test_accuracy']:.2%} | {r['test_two_recall']:.2%} | {r['test_two_precision']:.2%} | {r['test_macro_f1']:.3f} |")
text.extend(["","测试集两孔 53 张、三孔 2133 张。全判三孔可得 97.58% 准确率，应同时查看少数类指标。",
             "各实验的训练超参数见对应 config.json，训练曲线见 history.json。"
             "当前 GUI 使用原灰度模型；新模型导出到 deployment 后才启用组合分类。"])
text.extend(["","## 实际运行超参数","",
             "| 实验 | 初始学习率 | batch | weight decay | 两孔采样比例 | 类别权重幂 | 实际轮数 |",
             "| --- | ---: | ---: | ---: | --- | ---: | ---: |"])
for s in settings:
    fraction="自然分布" if s["fraction"] is None else f"{s['fraction']:.0%}"
    text.append(f"| {s['name']} | {s['lr']:g} | {s['batch']} | {s['wd']:g} | {fraction} | {s['weight']} | {s['epochs']} |")
text.extend(["","检查点均由验证集选出；测试结果已被查看，后续最终评估应另采集独立点位。",
             "本轮新增网络只替换孔数分类器，SEM 灰度图仍由原 U-Net 生成。"])
(output/"comparison.md").write_text("\n".join(text)+"\n",encoding="utf-8")
print("\n".join(text))
