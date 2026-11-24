"""Train, test or use a 256 px U-Net with original SEM grayscale targets."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
from torch.nn import functional as F

from dsa.data import read_mask
from dsa.engine import _checkpoint, json_file, pick_device
from dsa.metrics import components
from dsa.models.unet import PatternUNet
from dsa.preprocessing.pairs import SIZE, SPACING
from dsa.training.common import Pairs, _amp, _batches

MODEL_KIND = "sem_grayscale_256_v1"
CLASS_NAMES = ("2孔", "3孔")


def _means(value, mask):
    return (value*mask).flatten(1).sum(1)/mask.flatten(1).sum(1).clamp_min(1)


def _loss(pred, logits, target, c, valid, footprint, weights):
    error=(pred-target).abs()
    inside=(footprint>0.5)&valid
    outside=(~inside)&valid
    foreground=_means(error,inside).mean()
    background=_means(error,outside).mean()
    # Preserve hole contrast by comparing gradients inside the cavity.
    dy=(pred.diff(dim=-2)-target.diff(dim=-2)).abs()
    dx=(pred.diff(dim=-1)-target.diff(dim=-1)).abs()
    gy=_means(dy,inside[...,1:,:]&inside[...,:-1,:]).mean()
    gx=_means(dx,inside[...,:,1:]&inside[...,:,:-1]).mean()
    labelled=c!=-100
    losses=F.cross_entropy(logits,c,weight=weights,ignore_index=-100,reduction="none")
    class_loss=losses.sum()/(weights[c.clamp_min(0)]*labelled).sum().clamp_min(1)
    return .85*foreground + .15*background + .10*(gx+gy) + .12*class_loss


@torch.inference_mode()
def _evaluate(model,data,indices,device,batch_size,examples=None):
    model.eval()
    coordinate=torch.arange(SIZE,device=device)
    sums=torch.zeros(3,device=device)
    confusion=torch.zeros(4,dtype=torch.int64,device=device)
    saved=set()
    details=[] if examples is not None else None
    for indices_batch in _batches(indices,batch_size):
        x,y,c,valid=data.batch(indices_batch,device,coordinate)
        with _amp(device):
            masks,logits=model(x)
        pred=masks.float().sigmoid()
        error=(pred-y).abs()
        inside=(x[:,:1]>.5)&valid
        cell_error=_means(error,inside)
        full_error=_means(error,valid)
        mse=_means((pred-y).square(),inside)
        sums+=torch.stack((cell_error.sum(),full_error.sum(),mse.sum()))
        predicted=logits.argmax(1)
        eligible=c>=0
        codes=(2*c.clamp_min(0)+predicted)
        confusion.scatter_add_(0,codes,eligible.long())
        if examples is not None:
            probs=logits.float().softmax(1).cpu().numpy()
            errors=cell_error.cpu().tolist()
            for j,idx in enumerate(indices_batch):
                row=data.rows[idx]
                label=int(data.labels[idx])
                chosen=int(np.argmax(probs[j]))
                details.append(dict(cell_id=row["cell_id"],group_id=row["group_id"],
                                    truth=CLASS_NAMES[label] if label>=0 else "待复核",
                                    prediction=CLASS_NAMES[chosen],mae_template=errors[j],
                                    score_two=float(probs[j,0]),score_three=float(probs[j,1])))
                if label>=0 and label not in saved:
                    panel=Image.new("L",(3*SIZE,SIZE))
                    panel.paste(Image.fromarray(data.pixels[idx,0]),(0,0))
                    panel.paste(Image.fromarray(pred[j,0].mul(255).round().byte().cpu().numpy()),(SIZE,0))
                    panel.paste(Image.fromarray(data.pixels[idx,2]),(2*SIZE,0))
                    panel.save(examples/(row["cell_id"]+".png"))
                    saved.add(label)
    mae,full,mse=(sums/len(indices)).cpu().tolist()
    matrix=confusion.cpu().numpy().reshape(2,2)
    support=matrix.sum(1)
    f1=2*matrix.diagonal()/np.maximum(1,support+matrix.sum(0))
    macro=float(f1.mean())
    return dict(samples=len(indices),template_mae=mae,full_measured_mae=full,
                template_psnr_db=-10*math.log10(max(mse,1e-12)),macro_f1=macro,
                two_hole_recall=float(matrix[0,0]/max(1,support[0])),
                three_hole_recall=float(matrix[1,1]/max(1,support[1])),
                class_order=list(CLASS_NAMES),class_support=support.tolist(),
                confusion_matrix=matrix.tolist(),
                selection=mae+.15*(1-macro)), details


def _load_model(checkpoint,device):
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    return _model_from_state(state,device),state


def _model_from_state(state,device):
    if (state.get("model_kind"),state.get("size"),state.get("nm_per_px")) != (MODEL_KIND,SIZE,SPACING):
        raise ValueError("Checkpoint is not a 256 px SEM grayscale model")
    model=PatternUNet(state["width"],2).to(device=device,memory_format=torch.channels_last)
    model.load_state_dict(state["weights"])
    return model.eval()


def _write_test(model,data,device,args,out):
    out.mkdir(parents=True,exist_ok=False)
    examples=out/"examples"
    examples.mkdir()
    stats,details=_evaluate(model,data,data.indexes("test"),device,args.batch,examples)
    json_file(out/"metrics.json",stats)
    with (out/"predictions.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    print("TEST "+json.dumps(stats,ensure_ascii=False),flush=True)


def train(args):
    if args.epochs<=0 or args.batch<=0 or not math.isfinite(args.lr) or args.lr<=0:
        raise ValueError("Training epochs, batch and learning rate must be positive")
    if not 0 <= args.two_hole_fraction < 0.5:
        raise ValueError("Two-hole sampling fraction must be in [0, 0.5)")
    device=pick_device(args.device)
    torch.manual_seed(args.seed)
    data=Pairs(args.pairs)
    training,validation=data.indexes("train"),data.indexes("val")
    out=args.out
    out.mkdir(parents=True,exist_ok=False)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    json_file(out/"config.json",config | {"device_used":str(device),"size":SIZE,"nm_per_px":SPACING})
    model=PatternUNet(args.width,2).to(device=device,memory_format=torch.channels_last)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=args.lr/10)
    counts=np.bincount(data.labels[training][data.labels[training]>=0],minlength=2)
    if not counts.min():
        raise ValueError("Train partition must contain two-hole and three-hole cells")
    class_weights=1/np.sqrt(counts)
    class_weights=torch.tensor(class_weights/class_weights.mean(),dtype=torch.float32,device=device)
    coordinate=torch.arange(SIZE,device=device)
    rng=np.random.default_rng(args.seed)
    rare=training[data.labels[training]==0]
    desired=math.ceil((len(training)-len(rare))*args.two_hole_fraction/(1-args.two_hole_fraction))
    extra=max(0,desired-len(rare))
    epoch_size=len(training)+extra
    history=[]
    best=float("inf")
    print(f"Training on {device}: {len(training)} train, {len(validation)} val; class count {counts.tolist()}; extra two-hole draws per epoch: {extra}",flush=True)
    for epoch in range(1,args.epochs+1):
        model.train()
        running=torch.zeros((),device=device)
        sample_ids=np.concatenate((training,rng.choice(rare,size=extra,replace=True))) if extra else training
        for step,idx in enumerate(_batches(rng.permutation(sample_ids),args.batch),1):
            x,y,c,valid=data.batch(idx,device,coordinate)
            optimizer.zero_grad(set_to_none=True)
            with _amp(device):
                image,logits=model(x.contiguous(memory_format=torch.channels_last))
            loss=_loss(image.float().sigmoid(),logits.float(),y,c,valid,x[:,:1],class_weights)
            loss.backward()
            optimizer.step()
            running+=loss.detach()*len(idx)
            if step%75==0 or step==1:
                print(f"epoch={epoch} batch={step}/{math.ceil(epoch_size/args.batch)} loss={loss.item():.4f}",flush=True)
        scheduler.step()
        val,_=_evaluate(model,data,validation,device,args.batch)
        entry={"epoch":epoch,"train_loss":float(running.item()/epoch_size),"val":val}
        history.append(entry)
        json_file(out/"history.json",history)
        state=dict(model_kind=MODEL_KIND,size=SIZE,nm_per_px=SPACING,width=args.width,class_names=list(CLASS_NAMES),
                   weights=model.state_dict(),epoch=epoch,val=val,
                   test_sites=sorted({data.rows[i]["group_id"] for i in data.indexes("test")}))
        _checkpoint(out/"last.pt",state)
        if val["selection"]<best:
            best=val["selection"]
            _checkpoint(out/"best.pt",state)
        print("VALIDATION "+json.dumps(entry,ensure_ascii=False),flush=True)
    model,_=_load_model(out/"best.pt",device)
    _write_test(model,data,device,args,out/"test")
    print(f"Done: {out/'best.pt'}",flush=True)


@torch.inference_mode()
def predict(args):
    device=pick_device(args.device)
    model,state=_load_model(args.checkpoint,device)
    mask=read_mask(args.template)
    if mask.shape!=(SIZE,SIZE) or not mask.any():
        raise ValueError("Expected a nonempty 256 x 256 template mask at 0.78125 nm/px")
    sdf=ndi.distance_transform_edt(mask)-ndi.distance_transform_edt(~mask)
    inputs=np.stack((mask,np.clip(sdf*SPACING/24,-1,1)))
    predict_sem(model,state,device,inputs,args.out,review_score=.7)


@torch.inference_mode()
def predict_sem(model,state,device,inputs,out,review_score):
    # Match the training cache's 8-bit signed-distance encoding.
    inputs=inputs.astype(np.float32,copy=True)
    inputs[1]=np.rint((.5+.5*inputs[1])*255)/255*2-1
    x=torch.from_numpy(inputs)[None].to(device)
    pred,logits=model(x)
    classifier_name = "unet_joint"
    if "template_classifier" in state:
        from dsa.models.classifiers import build_classifier
        classifier_state = state["template_classifier"]
        classifier = build_classifier(classifier_state["model"]).to(device=device,memory_format=torch.channels_last)
        classifier.load_state_dict(classifier_state["weights"])
        classifier.eval()
        with _amp(device):
            logits = classifier(x.contiguous(memory_format=torch.channels_last))
        logits = logits.float()
        classifier_name = classifier_state["model"]
    scores=logits.softmax(1)[0].cpu().numpy()
    out.mkdir(parents=True,exist_ok=False)
    Image.fromarray(pred[0,0].sigmoid().mul(255).round().byte().cpu().numpy()).save(out/"predicted_sem.png")
    reasons=["sem_post_dsa_template_proxy"]
    if float(scores.max())<review_score:
        reasons.append("low_class_score")
    result=dict(model_kind=MODEL_KIND,classifier_model=classifier_name,prediction=CLASS_NAMES[int(scores.argmax())],
                class_scores_uncalibrated={k:float(v) for k,v in zip(CLASS_NAMES,scores)},
                needs_review=True,review_reasons=reasons,nm_per_px=SPACING,size=SIZE,
                checkpoint_epoch=state["epoch"])
    json_file(out/"prediction.json",result)
    return result


def _sem_dark_map(predicted, template):
    inner = ndi.distance_transform_edt(template) * SPACING >= 6
    smooth = ndi.gaussian_filter(predicted.astype(np.float32), sigma=1)
    if not inner.any():
        return np.zeros_like(template, dtype=bool), 0, smooth, inner, None
    baseline = float(np.median(smooth[inner]))
    radius = round(17 / SPACING)
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    local_background = ndi.grey_closing(smooth, footprint=x*x+y*y <= radius*radius)
    # Wall gradients alone can produce a large local contrast. Require darkness
    # relative to the cavity as well, and do not enforce the predicted class count.
    dark = (local_background - smooth >= 10) & (smooth <= baseline - 10)
    mask, regions = components(dark & inner, SPACING, min_area_nm2=20)
    return mask, len(regions), smooth, inner, baseline


def sem_preview_mask(predicted, template):
    """Estimate dark opening regions for display from an 8-bit model image."""
    mask, count, *_ = _sem_dark_map(predicted, template)
    return mask, count


def rank_sem_openings(predicted, template, target_count):
    """Select distinct dark cores and their contours without forcing weak candidates."""
    dark, count, smooth, inner, baseline = _sem_dark_map(predicted, template)
    selected = np.zeros_like(dark)
    if not dark.any():
        return dark, count, selected, baseline, []

    y, x = np.ogrid[:SIZE, :SIZE]
    window = 2 * round(5 / SPACING) + 1
    minima = dark & (smooth == ndi.minimum_filter(smooth, size=window))
    labels, n = ndi.label(minima, structure=np.ones((3, 3), dtype=bool))
    candidates = []
    for index in range(1, n+1):
        sy, sx = np.nonzero(labels == index)
        mid = np.argmin((sy-sy.mean())**2+(sx-sx.mean())**2)
        cy, cx = int(sy[mid]), int(sx[mid])
        r2 = ((x-cx)*SPACING)**2+((y-cy)*SPACING)**2
        ring = inner & (r2 >= 10**2) & (r2 <= 18**2)
        if not ring.any():
            continue
        core = float(smooth[r2 <= 3**2].mean())
        local = float(np.median(smooth[ring]))
        if local-core >= 6 and baseline-core >= 10:
            candidates.append((cx, cy, core, local, r2))
    # Rank by the grayscale difference from the same cavity reference.
    candidates.sort(key=lambda c: (c[2], c[1], c[0]))
    seeds = []
    for item in candidates:
        cx, cy, *_ = item
        if all(((cx-px)**2+(cy-py)**2)*SPACING**2 >= 16**2 for px,py,*_ in seeds):
            seeds.append(item)

    details = []
    if seeds:
        owner = np.argmin(np.stack([r2 for *_, r2 in seeds]), axis=0)
        for index, (cx, cy, core, local, r2) in enumerate(seeds):
            # Leave a narrow gap between neighboring basins in a merged yellow region.
            basin = ndi.binary_erosion(owner == index) if len(seeds)>1 else owner == index
            contour = (dark & basin & (r2 <= 15**2) & (smooth <= core + .65*(local-core)))
            regions, _ = ndi.label(contour, structure=np.ones((3, 3), dtype=bool))
            seed_label = regions[cy,cx]
            if seed_label == 0:
                continue
            shape = regions == seed_label
            area = float(shape.sum()*SPACING**2)
            if area < 20:
                continue
            picked = len(details) < target_count
            if picked:
                selected |= shape
            details.append(dict(rank=len(details)+1, selected=picked, center_xy_px=[cx,cy],
                                core_gray=round(core,2), darkness_gray_levels=round(baseline-core,2),
                                local_contrast_gray_levels=round(local-core,2), area_nm2=round(area,2)))
    return dark, count, selected, baseline, details


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    fit=sub.add_parser("train")
    fit.add_argument("--pairs",type=Path,required=True)
    fit.add_argument("--epochs",type=int,default=8)
    fit.add_argument("--batch",type=int,default=16)
    fit.add_argument("--width",type=int,default=8)
    fit.add_argument("--two-hole-fraction",type=float,default=0.0,
                     help="minimum train-epoch fraction via resampling; validation/test are unchanged")
    fit.add_argument("--lr",type=float,default=.001)
    fit.add_argument("--seed",type=int,default=42)
    infer=sub.add_parser("predict")
    infer.add_argument("--template",type=Path,required=True)
    infer.add_argument("--checkpoint",type=Path,required=True)
    for command in (fit,infer):
        command.add_argument("--out",type=Path,required=True)
        command.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
        command.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.command=="train":
        train(args)
    else:
        predict(args)


if __name__=="__main__":
    main()
