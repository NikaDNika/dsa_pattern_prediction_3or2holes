"""Compare classifiers using only template geometry and fixed physical-site splits."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dsa.models.classifiers import build_classifier
from dsa.training.common import Pairs, _amp, _batches
from dsa.engine import _checkpoint, json_file, pick_device


def metrics(y,p):
    predicted=p.argmax(1)
    cm=np.bincount(2*y+predicted,minlength=4).reshape(2,2)
    tp,fn=cm[0];fp,tn=cm[1]
    f1=2*cm.diagonal()/np.maximum(1,cm.sum(0)+cm.sum(1))
    return dict(accuracy=float((tp+tn)/len(y)),macro_f1=float(f1.mean()),
                balanced_accuracy=float((tp/max(1,tp+fn)+tn/max(1,tn+fp))/2),
                two_recall=float(tp/max(1,tp+fn)),two_precision=float(tp/max(1,tp+fp)),
                three_recall=float(tn/max(1,tn+fp)),confusion_matrix=cm.tolist(),
                samples=len(y),class_order=["2孔","3孔"])


def inputs(data,ids,device):
    a=torch.from_numpy(data.pixels[ids,:2]).to(device).float()/255
    return torch.cat((a[:,:1],a[:,1:2]*2-1),1).contiguous(memory_format=torch.channels_last)


@torch.inference_mode()
def evaluate(model,data,ids,device,batch):
    model.eval()
    probs=[]
    for chunk in _batches(ids,batch):
        with _amp(device):
            logits=model(inputs(data,chunk,device))
        probs.append(logits.float().softmax(1).cpu().numpy())
    p=np.concatenate(probs)
    return metrics(data.labels[ids],p),p


def train_one(cfg,data,out,device,seed):
    torch.manual_seed(seed);np.random.seed(seed)
    rng=np.random.default_rng(seed)
    out.mkdir(parents=True,exist_ok=False)
    model=build_classifier(cfg["model"]).to(device=device,memory_format=torch.channels_last)
    (out/"architecture.txt").write_text(str(model)+"\n",encoding="utf-8")
    optim=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,cfg["epochs"],eta_min=cfg["lr"]*.05)
    parts={s:np.array([i for i in data.indexes(s) if data.labels[i]>=0]) for s in ("train","val","test")}
    counts=np.bincount(data.labels[parts["train"]],minlength=2)
    if counts.min()==0:
        raise ValueError("Both two-hole and three-hole training samples are required")
    weight=(counts.sum()/(2*counts))**cfg["weight_power"]
    weights=torch.tensor(weight,dtype=torch.float32,device=device)
    best=-1;history=[];stale=0
    json_file(out/"config.json",cfg|dict(seed=seed,parameters=sum(p.numel() for p in model.parameters()),
        size=256,nm_per_px=.78125,pretrained=False,train_class_counts=counts.tolist()))
    start=time.time()
    for epoch in range(1,cfg["epochs"]+1):
        model.train();loss_sum=0
        fraction=cfg.get("two_hole_fraction")
        if fraction is None:
            order=rng.permutation(parts["train"])
        else:
            if not 0 < fraction < 1:
                raise ValueError("two_hole_fraction must be between zero and one")
            # Balance the sampled labels, without changing validation/test prevalence.
            pools=[parts["train"][data.labels[parts["train"]]==c] for c in (0,1)]
            n_two=round(len(parts["train"])*fraction)
            order=np.concatenate([rng.choice(pools[0],n_two,replace=True),
                                  rng.choice(pools[1],len(parts["train"])-n_two,replace=True)])
            rng.shuffle(order)
        for batch,ids in enumerate(_batches(order,cfg["batch"]),1):
            x=inputs(data,ids,device)
            if cfg["augment"]:
                if rng.random()<.5:x=x.flip(-1)
                if rng.random()<.5:x=x.flip(-2)
            y=torch.from_numpy(data.labels[ids]).to(device)
            optim.zero_grad(set_to_none=True)
            with _amp(device):
                logits=model(x)
                loss=F.cross_entropy(logits,y,weight=weights,label_smoothing=cfg["label_smoothing"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5)
            optim.step()
            loss_sum+=loss.item()*len(ids)
            if batch==1 or batch%150==0:
                print(out.name,epoch,batch,"loss",round(loss.item(),4),flush=True)
        learning_rate=optim.param_groups[0]["lr"]
        scheduler.step()
        val,_=evaluate(model,data,parts["val"],device,cfg["batch"])
        history.append(dict(epoch=epoch,lr=learning_rate,train_loss=loss_sum/len(order),validation=val))
        json_file(out/"history.json",history)
        if val["macro_f1"]>best:
            best=val["macro_f1"];stale=0
            _checkpoint(out/"best.pt",dict(model_kind="template_classifier_v1",model=cfg["model"],
                 weights=model.state_dict(),epoch=epoch,size=256,nm_per_px=.78125,classes=["2孔","3孔"],
                 selection="validation_macro_f1",validation=val))
        else:stale+=1
        print(out.name,"VALIDATION",epoch,json.dumps(val),flush=True)
        if epoch>=cfg.get("min_epochs",1) and stale>=cfg["patience"]:break
    state=torch.load(out/"best.pt",map_location=device,weights_only=True)
    model.load_state_dict(state["weights"])
    result=dict(config=cfg,best_epoch=state["epoch"],validation=state["validation"],seconds=time.time()-start)
    for split in ("train","test"):
        score,p=evaluate(model,data,parts[split],device,cfg["batch"])
        result[split]=score
        np.savez_compressed(out/(split+"_scores.npz"),indices=parts[split],scores=p,truth=data.labels[parts[split]])
    json_file(out/"metrics.json",result)
    print(out.name,"DONE",json.dumps(result),flush=True)
    del model,optim
    torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config",type=Path,help="train all runs defined in a JSON config")
    mode.add_argument("--checkpoint",type=Path,help="evaluate a saved standalone classifier")
    p.add_argument("--pairs",type=Path,default=Path("data/sem_pairs_256"))
    p.add_argument("--split",choices=("train","val","test"),default="test")
    p.add_argument("--out",type=Path,help="new output directory for checkpoint evaluation")
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    p.add_argument("--batch",type=int,default=24)
    a=p.parse_args()
    torch.set_num_threads(4)
    if a.checkpoint:
        if a.out is None:p.error("--out is required with --checkpoint")
        if a.batch<=0:p.error("--batch must be positive")
        device=pick_device(a.device)
        state=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
        if (state.get("model_kind"),state.get("size"),state.get("nm_per_px")) != ("template_classifier_v1",256,.78125):
            raise ValueError("Expected a standalone 256-pixel template classifier checkpoint")
        model=build_classifier(state["model"]).to(device=device,memory_format=torch.channels_last)
        model.load_state_dict(state["weights"])
        data=Pairs(a.pairs)
        ids=np.array([i for i in data.indexes(a.split) if data.labels[i]>=0])
        score,probs=evaluate(model,data,ids,device,a.batch)
        a.out.mkdir(parents=True,exist_ok=False)
        json_file(a.out/"metrics.json",dict(checkpoint=str(a.checkpoint),split=a.split,metrics=score))
        np.savez_compressed(a.out/"scores.npz",indices=ids,scores=probs,truth=data.labels[ids])
        print(json.dumps(score,ensure_ascii=False),flush=True)
        return
    config=json.loads(a.config.read_text(encoding="utf-8"))
    names=set()
    for run in config["runs"]:
        name=run["name"]
        if not name or Path(name).name!=name or name in names:
            raise ValueError("Run names must be unique directory names")
        names.add(name)
        if run["epochs"]<=0 or run["batch"]<=0 or run["patience"]<=0:
            raise ValueError("epochs, batch and patience must be positive")
        if not math.isfinite(run["lr"]) or run["lr"]<=0 or run["weight_decay"]<0:
            raise ValueError("lr must be positive and finite; weight_decay must be nonnegative")
        if not 0<=run["label_smoothing"]<1 or run["weight_power"]<0:
            raise ValueError("Invalid label_smoothing or weight_power")
        if not 1<=run.get("min_epochs",1)<=run["epochs"]:
            raise ValueError("min_epochs must be between one and epochs")
        if run.get("two_hole_fraction") is not None and not 0<run["two_hole_fraction"]<1:
            raise ValueError("two_hole_fraction must be between zero and one")
    if not names:raise ValueError("At least one run is required")
    device=pick_device(config.get("device","auto"))
    data=Pairs(Path(config["pairs"]))
    out=Path(config["out"]);out.mkdir(parents=True,exist_ok=False)
    json_file(out/"experiment_config.json",config)
    results=[]
    for run in config["runs"]:
        results.append(train_one(run,data,out/run["name"],device,config["seed"]))
        json_file(out/"comparison.json",results)
    chosen=max(results,key=lambda r:r["validation"]["macro_f1"])
    json_file(out/"selected.json",dict(name=chosen["config"]["name"],
              rule="maximum validation macro F1; test not used for selection"))
    print("SELECTED",chosen["config"]["name"],flush=True)


if __name__=="__main__":main()
