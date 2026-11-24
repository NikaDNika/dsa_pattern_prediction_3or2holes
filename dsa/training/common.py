"""Shared paired SEM dataset, batching and precision policy for training flows."""
import numpy as np
import torch
from dsa.preprocessing.pairs import SIZE, _read


class Pairs:
    def __init__(self, root):
        self.rows = _read(root/"pairs.csv")
        self.pixels = np.load(root/"paired_pixels.npy",mmap_mode="r",allow_pickle=False)
        if self.pixels.shape!=(len(self.rows),3,SIZE,SIZE) or self.pixels.dtype!=np.uint8:
            raise ValueError("Pair cache dimensions do not match the manifest")
        partitions = {}
        for i,row in enumerate(self.rows):
            if int(row["index"])!=i:
                raise ValueError("Unexpected pair cache index")
            split = row["split"]
            group = row["group_id"]
            if split!="excluded_truncated" and partitions.setdefault(group,split)!=split:
                raise ValueError(f"Physical site shared between partitions: {group}")
        self.labels = np.array([int(r["label"]) for r in self.rows],dtype=np.int64)
        self.valid = np.array([[int(r[k]) for k in ("valid_x0","valid_y0","valid_x1","valid_y1")]
                              for r in self.rows],dtype=np.int16)

    def indexes(self, split):
        indices=np.array([i for i,r in enumerate(self.rows) if r["split"]==split],dtype=np.int64)
        if not len(indices):
            raise ValueError(f"Empty {split} partition")
        return indices

    def batch(self, indices, device, coordinate):
        values=torch.from_numpy(self.pixels[indices]).to(device).float().mul_(1/255)
        image=values[:,:2]
        image=torch.cat((image[:,:1], image[:,1:2]*2-1),dim=1)
        y=values[:,2:3]
        c=torch.from_numpy(self.labels[indices]).to(device)
        bounds=torch.from_numpy(self.valid[indices]).to(device)
        x0,y0,x1,y1=(bounds[:,i,None,None] for i in range(4))
        xx=coordinate[None,None,:]
        yy=coordinate[None,:,None]
        valid=((xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)).unsqueeze(1)
        return image,y,c,valid


def _batches(indices, n):
    for a in range(0,len(indices),n):
        yield indices[a:a+n]


def _amp(device):
    return torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda")
