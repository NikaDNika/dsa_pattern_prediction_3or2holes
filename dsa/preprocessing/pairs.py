"""Build 256 px template/SEM pairs from the 200 nm dataset and original TIFFs."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import re
import tempfile

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from dsa.preprocessing.segmentation import _otsu

SIZE = 256
SPACING = 200 / SIZE
LABELS = {"2孔": 0, "3孔": 1}
SPLITS = ("train", "val", "test")


def _read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _site(source):
    # Acquisition sequence and Windows copy suffix are not part of the physical site ID.
    match = re.fullmatch(r"(.+-\d+-\d+)-\d+(?: \(\d+\))?", Path(source).stem)
    if not match:
        raise ValueError(f"Cannot derive physical site from {source}")
    return match[1]


def _assign_groups(rows, seed):
    sites = sorted({r["group_id"] for r in rows if r["class_folder"] != "边缘截断"})
    if len(sites) < 4:
        raise ValueError("Need at least four physical sites for train/val/test")
    site_i = {name: i for i, name in enumerate(sites)}
    scales = sorted({r["source_nm_per_px"] for r in rows if r["class_folder"] != "边缘截断"})
    proportions = np.array([.70, .15, .15])
    counts = np.zeros((len(sites), 4 + len(scales)), dtype=int)
    for r in rows:
        if r["class_folder"] == "边缘截断":
            continue
        index = site_i[r["group_id"]]
        counts[index, LABELS.get(r["class_folder"], 2)] += 1
        counts[index, 3] += 1
        counts[index, 4 + scales.index(r["source_nm_per_px"])] += 1
    rng = np.random.default_rng(seed)
    n_hold = max(1, round(.18 * len(sites)))
    best = None
    for _ in range(2500):
        parts = np.split(rng.permutation(len(sites)), [len(sites) - 2*n_hold, len(sites) - n_hold])
        sizes = np.array([counts[part].sum(axis=0) for part in parts])
        # Preserve both labels in every split and all acquisition scales in training.
        priority = (int((sizes[:, :2] == 0).sum()), int((sizes[0, 4:] == 0).sum()))
        ratio = sizes / np.maximum(counts.sum(axis=0), 1)
        rank = (*priority, float(abs(ratio[:, :4] - proportions[:, None]).sum()))
        if best is None or rank < best:
            best, chosen = rank, parts
    if best[0]:
        raise ValueError("Cannot produce a site-separated split containing both labels in each set")
    return {sites[i]: split for split, part in zip(SPLITS, chosen) for i in part}


def _center(row):
    return float(row["center_x_px"]), float(row["center_y_px"])


def _load_component_index(smooth, spacing, edge, requested, typical=None):
    if edge:
        background = ndi.uniform_filter(smooth, size=max(3, round(213 / spacing)))
    else:
        background = ndi.gaussian_filter(smooth, 63 / spacing)
    contrast = (smooth - background) / 255 + .5
    labels, _ = ndi.label(contrast < _otsu(contrast))
    H, W = labels.shape
    margin = max(2, round(12 / spacing))
    index = {}
    for label_id, slices in enumerate(ndi.find_objects(labels), 1):
        if slices is None:
            continue
        sy,sx = slices
        l,t,r,b = sx.start,sy.start,sx.stop,sy.stop
        if edge:
            if not (l<=2 or t<=2 or r>=W-2 or b>=H-2):
                continue
            if not (23<=(r-l)*spacing<=107 and 50<=(b-t)*spacing<=205):
                continue
            x0,y0,x1,y1 = max(0,l-margin),max(0,t-margin),min(W,r+margin),min(H,b+margin)
            tw,th = typical
            l = min(0,x1-margin-round(tw)) if x0==0 else x0+margin
            t = min(0,y1-margin-round(th)) if y0==0 else y0+margin
            r = max(W,x0+margin+round(tw)) if x1==W else x1-margin
            b = max(H,y0+margin+round(th)) if y1==H else y1-margin
        else:
            if not (31<=(r-l)*spacing<=106 and 76<=(b-t)*spacing<=206):
                continue
            if l<=1 or t<=1 or r>=W-1 or b>=H-1:
                continue
        center=(round((l+r-1)/2,3),round((t+b-1)/2,3))
        if center in requested:
            if center in index:
                raise ValueError(f"Ambiguous detector center: {center}")
            index[center] = label_id,slices
    if index.keys() != requested:
        raise ValueError(f"{len(requested - index.keys())} unmatched source centers")
    return labels, index


def _component_mask(labels, match, row, source_spacing):
    label_id, slices = match
    sy,sx = slices
    # The mask describes the footprint only. The dark pore intensities never enter the input.
    contour = ndi.binary_fill_holes(labels[slices] == label_id)
    source = Image.fromarray(contour.astype(np.uint8) * 255)
    left = float(row["crop_left_edge_px"])
    top = float(row["crop_top_edge_px"])
    step = SPACING / source_spacing
    mask = np.asarray(source.transform((SIZE,SIZE), Image.Transform.AFFINE,
        (step,0,left-sx.start,0,step,top-sy.start), Image.Resampling.NEAREST, fillcolor=0)) > 0
    if not mask.any():
        raise ValueError(f"Empty template {row['cell_id']}")
    return mask


def _valid_bounds(row, shape, source_spacing):
    H,W = shape
    step = SPACING / source_spacing
    x = float(row["crop_left_edge_px"]) + (np.arange(SIZE)+.5)*step
    y = float(row["crop_top_edge_px"]) + (np.arange(SIZE)+.5)*step
    ix,iy = np.flatnonzero((x>=0)&(x<W)), np.flatnonzero((y>=0)&(y<H))
    return int(ix[0]),int(iy[0]),int(ix[-1]+1),int(iy[-1]+1)


def prepare(args):
    rows = _read(args.images / "统一比例尺单元清单.csv")
    if len({r["cell_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate cells in the unified manifest")
    by_source = defaultdict(list)
    for index,row in enumerate(rows):
        if (float(row["field_width_nm"]),float(row["field_height_nm"]),float(row["output_nm_per_px"])) != (200,200,SPACING):
            raise ValueError("Expected 200 nm views at 256 x 256")
        row["index"] = index
        row["group_id"] = _site(row["source_relative_path"])
        by_source[row["source_relative_path"]].append(row)
    assignments = _assign_groups(rows,args.seed)
    for row in rows:
        row["label"] = LABELS.get(row["class_folder"], -100)
        row["split"] = "excluded_truncated" if row["class_folder"]=="边缘截断" else assignments[row["group_id"]]

    out = args.out.resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    images = args.images.resolve()
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        stage = Path(temp)/"pairs"
        stage.mkdir()
        (stage/"templates").mkdir()
        (stage/"examples").mkdir()
        cache = np.lib.format.open_memmap(stage/"paired_pixels.npy",mode="w+",dtype=np.uint8,
                                         shape=(len(rows),3,SIZE,SIZE))

        def source_batch(entry):
            relative,items = entry
            spacing = float(items[0]["source_nm_per_px"])
            with Image.open(args.raw/relative) as im:
                raw = np.asarray(im.crop((0,0,im.width,im.height*729//768)).convert("L"),dtype=np.float32)
            smooth = ndi.gaussian_filter(raw,3/spacing)
            label_sets = {}
            typical = None
            for edge in (False,True):
                wanted = {_center(r) for r in items if (r["class_folder"]=="边缘截断")==edge}
                if wanted:
                    label_sets[edge] = _load_component_index(smooth,spacing,edge,wanted,typical)
                    if not edge:
                        typical=np.median([(x.stop-x.start,y.stop-y.start)
                                           for _,(y,x) in label_sets[edge][1].values()],axis=0)
            for r in items:
                edge = r["class_folder"]=="边缘截断"
                labels,matches = label_sets[edge]
                mask = _component_mask(labels,matches[_center(r)],r,spacing)
                idx = r["index"]
                cache[idx,0] = mask*255
                distance = ndi.distance_transform_edt(mask)-ndi.distance_transform_edt(~mask)
                cache[idx,1] = np.rint((.5+.5*np.clip(distance*SPACING/24,-1,1))*255).astype(np.uint8)
                with Image.open(images/r["output_png"]) as target:
                    if target.size!=(SIZE,SIZE) or target.mode!="L":
                        raise ValueError(f"Bad label image {r['output_png']}")
                    cache[idx,2] = np.asarray(target)
                template = Path("templates")/(r["cell_id"]+".png")
                Image.fromarray(cache[idx,0]).save(stage/template,compress_level=1)
                r["template_path"] = template.as_posix()
                r["label_image_path"] = str(images/r["output_png"])
                r["valid_x0"],r["valid_y0"],r["valid_x1"],r["valid_y1"] = _valid_bounds(r,raw.shape,spacing)
            print(f"prepared {items[0]['source_id']} {len(items)} pairs",flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(source_batch,by_source.items()))
        for cell in ("F048_R02_C18","F048_R03_C09","F001_R06_C10"):
            example = next((r for r in rows if r["cell_id"]==cell),None)
            if example:
                i=example["index"]
                panel=Image.new("L",(SIZE*2,SIZE))
                panel.paste(Image.fromarray(cache[i,0]),(0,0))
                panel.paste(Image.fromarray(cache[i,2]),(SIZE,0))
                panel.save(stage/"examples"/(cell+".png"))
        cache.flush()
        del cache
        _write(stage/"pairs.csv",rows)
        stats=Counter((r["split"],r["class_folder"]) for r in rows)
        _write(stage/"split_summary.csv",[dict(split=split,label=label,count=count)
                                          for (split,label),count in sorted(stats.items())])
        stage.rename(out)
    print(f"Prepared {len(rows)} paired cells in {out}",flush=True)
    for (split,label),count in sorted(stats.items()):
        print(split,label,count,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw",type=Path,required=True)
    p.add_argument("--images",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--workers",type=int,default=6)
    prepare(p.parse_args())


if __name__=="__main__":
    main()
