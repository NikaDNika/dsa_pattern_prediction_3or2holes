"""Export the validation-selected classifier with the existing SEM generator."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from dsa.engine import _checkpoint,json_file

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--comparison",type=Path,required=True)
p.add_argument("--sem",type=Path,required=True)
p.add_argument("--out",type=Path,required=True)
args=p.parse_args()
selected=json.loads((args.comparison/"selected.json").read_text(encoding="utf-8"))
chosen=args.comparison/selected["name"]/"best.pt"
classifier=torch.load(chosen,map_location="cpu",weights_only=True)
sem=torch.load(args.sem,map_location="cpu",weights_only=True)
if (classifier.get("model_kind"),classifier.get("size"),classifier.get("nm_per_px")) != ("template_classifier_v1",256,.78125):
    raise ValueError("Expected a standalone 256-pixel template classifier")
if (sem.get("model_kind"),sem.get("size"),sem.get("nm_per_px")) != ("sem_grayscale_256_v1",256,.78125):
    raise ValueError("Expected a compatible 256-pixel grayscale generator")
if classifier["validation"]["macro_f1"] <= sem["val"]["macro_f1"]:
    raise ValueError("The selected classifier did not improve validation macro F1")
if min(classifier["validation"]["two_recall"],classifier["validation"]["three_recall"])<=0:
    raise ValueError("The selected classifier failed to recognize one validation class")
args.out.mkdir(parents=True,exist_ok=False)
_checkpoint(args.out/"best.pt",sem|{"template_classifier":classifier})
json_file(args.out/"selection.json",dict(
    classifier=chosen.as_posix(),generator=args.sem.as_posix(),
    criterion="validation macro F1",classifier_validation=classifier["validation"],
    original_validation=sem["val"]))
print(args.out/"best.pt")
