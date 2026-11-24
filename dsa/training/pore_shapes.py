"""Fit a separate pore-shape U-Net using edge-derived SEM proxy labels."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.signal import find_peaks
import torch
from torch.nn import functional as F

from dsa.engine import _checkpoint, json_file, pick_device
from dsa.metrics import components
from dsa.models.unet import PatternUNet
from dsa.preprocessing.pairs import SIZE, SPACING
from dsa.training.reconstruction import MODEL_KIND
from dsa.training.common import Pairs, _batches, _amp

SHAPE_KIND = "sem_pore_shape_v1"
LABEL_PROVENANCE = "estimated_radial_sem_edges"


def trace_sem_pores(image, cavity, expected_count):
    """Trace outward intensity transitions from independently detected pore centers."""
    ys, xs = np.nonzero(cavity)
    if len(ys) == 0:
        return None
    y0,y1 = int(ys.min()), int(ys.max()) + 1
    x0,x1 = int(xs.min()), int(xs.max()) + 1
    if y1-y0 < 1.4*(x1-x0):
        return None
    smooth = ndi.gaussian_filter(image.astype(np.float32), sigma=1.7)
    center_x = (x0+x1-1)//2
    half_width = round(6 / SPACING)
    profile = smooth[y0:y1, center_x-half_width:center_x+half_width+1].mean(axis=1)
    peaks, _ = find_peaks(-profile, prominence=8, distance=round(21/SPACING))
    peaks = [p+y0 for p in peaks if .13 < p/len(profile) < .87]
    if len(peaks) != expected_count:
        return None

    angles = np.arange(64)*2*np.pi/64
    radius = np.arange(0, 18/SPACING + .5, .5)
    ray_dx = np.cos(angles)[:, None]
    ray_dy = np.sin(angles)[:, None]
    boundary = Image.new("L", (SIZE,SIZE))
    draw = ImageDraw.Draw(boundary)
    safety = ndi.distance_transform_edt(cavity)*SPACING >= 3
    for center_y in peaks:
        region = smooth[center_y-3:center_y+4, center_x-half_width:center_x+half_width+1]
        jy,jx = np.unravel_index(region.argmin(),region.shape)
        cy,cx = center_y+jy-3,center_x+jx-half_width
        if not safety[cy,cx]:
            return None
        yy = cy+ray_dy*radius
        xx = cx+ray_dx*radius
        levels = ndi.map_coordinates(smooth, (yy,xx),order=1)
        in_cavity = ndi.map_coordinates(safety.astype(np.uint8),(yy,xx),order=0)>0
        gradient = np.gradient(ndi.gaussian_filter1d(levels,2,axis=1),axis=1)
        search = (radius[None,:]*SPACING>=4.5) & in_cavity
        ridge = np.argmax(np.where(search,gradient,-np.inf),axis=1)
        prominence = levels[np.arange(len(angles)),ridge] - levels[:,0]
        accepted = search[np.arange(len(angles)),ridge] & (prominence>=4)
        if accepted.mean()<.65:
            return None
        distance = np.interp(angles,angles[accepted],radius[ridge[accepted]],period=2*np.pi)
        distance = ndi.gaussian_filter1d(distance,1.3,mode="wrap")
        draw.polygon(list(zip(cx+np.cos(angles)*distance,cy+np.sin(angles)*distance)), fill=255)
    mask = (np.asarray(boundary)>0)&safety
    mask, regions = components(mask,SPACING,min_area_nm2=20)
    if len(regions)!=expected_count:
        return None
    return mask


def build_targets(pairs,out):
    mask_dir=out/"label_examples"
    mask_dir.mkdir()
    archive=np.lib.format.open_memmap(out/"pore_edge_labels.npy",mode="w+",dtype=np.uint8,
                                       shape=(len(pairs.rows),SIZE,SIZE))
    usable={split:[] for split in ("train","val","test")}
    counts=Counter()
    examples=Counter()
    for i,row in enumerate(pairs.rows):
        if row["split"] not in usable or pairs.labels[i]<0:
            continue
        target=trace_sem_pores(pairs.pixels[i,2],pairs.pixels[i,0]>0,int(pairs.labels[i])+2)
        if target is None:
            continue
        archive[i]=target
        usable[row["split"]].append(i)
        key=(row["split"],int(pairs.labels[i])+2)
        counts[key]+=1
        if examples[key]<2:
            reference=np.stack([pairs.pixels[i,2]]*3,axis=-1)
            edge=target&~ndi.binary_erosion(target)
            reference[edge]=(255,35,30)
            Image.fromarray(reference).save(mask_dir/(row["cell_id"]+".png"))
            examples[key]+=1
        if i%1500==0:
            print(f"Contour labels checked: {i}/{len(pairs.rows)}",flush=True)
    archive.flush()
    for split in usable:
        if not all(counts[(split,c)] for c in (2,3)):
            raise ValueError(f"Contour targets for both pore counts are required in {split}")
    json_file(out/"label_summary.json",{f"{s}_{n}孔":counts[s,n] for s,n in sorted(counts)})
    np.savez(out/"label_indices.npz",**{k:np.asarray(v) for k,v in usable.items()})
    print("Proxy contour coverage",counts,flush=True)
    return archive,{k:np.asarray(v) for k,v in usable.items()}


def shape_from_state(state,device):
    if state.get("kind")!=SHAPE_KIND:
        raise ValueError("Invalid pore shape checkpoint")
    model=PatternUNet(state["width"],2).to(device=device,memory_format=torch.channels_last)
    model.load_state_dict(state["weights"])
    return model.eval()


@torch.inference_mode()
def predict_shape(state,inputs,device):
    x=inputs.astype(np.float32,copy=True)
    x[1]=np.rint((.5+.5*x[1])*255)/255*2-1
    model=shape_from_state(state,device)
    logits,_=model(torch.from_numpy(x)[None].to(device))
    mask=(logits[0,0].sigmoid().cpu().numpy()>=state["threshold"])&(inputs[0]>0)
    return components(mask,SPACING,min_area_nm2=20)


def _shape_loss(logits,target,cavity):
    pred=logits.sigmoid()
    flat=lambda t:t.flatten(1)
    dice=(2*flat(pred*target).sum(1)+1)/(flat(pred+target).sum(1)+1)
    weights=1+2*target+cavity
    bce=(F.binary_cross_entropy_with_logits(logits,target,reduction="none")*weights).mean()
    return bce+(1-dice.mean())


@torch.inference_mode()
def _score(model,pairs,masks,indices,device,batch_size,coordinate):
    model.eval()
    sum_dice=torch.zeros(2,device=device)
    count=torch.zeros(2,device=device)
    for selected in _batches(indices,batch_size):
        x,_,classes,_=pairs.batch(selected,device,coordinate)
        target=torch.from_numpy(masks[selected]).to(device)>0
        with _amp(device):
            logits,_=model(x.contiguous(memory_format=torch.channels_last))
        pred=(logits[:,0]>0)&(x[:,0]>.5)
        intersection=(pred&target).flatten(1).sum(1)
        dice=2*intersection/(pred.flatten(1).sum(1)+target.flatten(1).sum(1)).clamp_min(1)
        sum_dice.scatter_add_(0,classes,dice.float())
        count.scatter_add_(0,classes,torch.ones_like(dice))
    by_class=(sum_dice/count).cpu().tolist()
    return {"proxy_dice_two":by_class[0],"proxy_dice_three":by_class[1],
            "mean_class_proxy_dice":sum(by_class)/2,"class_counts":count.int().cpu().tolist()}


def train(args):
    if args.epochs<=0 or args.batch<=0:
        raise ValueError("Epochs and batch size must be positive")
    device=pick_device(args.device)
    torch.manual_seed(42)
    rng=np.random.default_rng(42)
    data=Pairs(args.pairs)
    base=torch.load(args.sem_checkpoint,map_location="cpu",weights_only=True)
    if (base.get("model_kind"),base.get("size"),base.get("nm_per_px"))!=(MODEL_KIND,SIZE,SPACING):
        raise ValueError("Base checkpoint must be from 256 px SEM grayscale training")
    test_sites=sorted({r["group_id"] for r in data.rows if r["split"]=="test"})
    if base.get("test_sites")!=test_sites:
        raise ValueError("The base model and pore-shape data must use the same held-out sites")
    out=args.out
    out.mkdir(parents=True,exist_ok=False)
    masks,split=build_targets(data,out)
    coordinate=torch.arange(SIZE,device=device)
    model=PatternUNet(8,2).to(device=device,memory_format=torch.channels_last)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=.00015)
    train_ids=split["train"]
    rare=train_ids[data.labels[train_ids]==0]
    extra=max(0,math.ceil((len(train_ids)-len(rare))*.18/.82)-len(rare))
    history=[]
    best=-1
    for epoch in range(1,args.epochs+1):
        model.train()
        order=np.concatenate((train_ids,rng.choice(rare,size=extra,replace=True)))
        for batch,i in enumerate(_batches(rng.permutation(order),args.batch),1):
            x,_,_,_=data.batch(i,device,coordinate)
            target=torch.from_numpy(masks[i]).to(device).float().unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            with _amp(device):
                logits,_=model(x.contiguous(memory_format=torch.channels_last))
            loss=_shape_loss(logits.float(),target,x[:,:1])
            loss.backward()
            optimizer.step()
            if batch==1 or batch%100==0:
                print(f"Pore shape epoch {epoch}, batch {batch}, loss {loss.item():.4f}",flush=True)
        scheduler.step()
        v=_score(model,data,masks,split["val"],device,args.batch,coordinate)
        history.append({"epoch":epoch,"validation":v})
        json_file(out/"history.json",history)
        if v["mean_class_proxy_dice"]>best:
            best=v["mean_class_proxy_dice"]
            state=dict(kind=SHAPE_KIND,label_provenance=LABEL_PROVENANCE,width=8,
                       weights=model.state_dict(),threshold=.5,epoch=epoch,val=v)
            _checkpoint(out/"best.pt",base|{"hole_shape":state})
        print("VALIDATION",v,flush=True)
    state=torch.load(out/"best.pt",map_location="cpu",weights_only=True)["hole_shape"]
    test=_score(shape_from_state(state,device),data,masks,split["test"],device,args.batch,coordinate)
    json_file(out/"test_metrics.json",test)
    print("TEST (against estimated SEM edges)",test,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs",type=Path,required=True)
    p.add_argument("--sem-checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=4)
    p.add_argument("--batch",type=int,default=24)
    p.add_argument("--device",choices=("cpu","cuda","auto"),default="auto")
    torch.set_num_threads(4)
    train(p.parse_args())


if __name__=="__main__":
    main()
