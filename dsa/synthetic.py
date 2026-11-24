"""Generate correlated, deliberately nonphysical examples for pipeline smoke tests."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .data import CLASSES, Sample, write_manifest


def make_demo(root, groups, cells, seed):
    if groups < 6 or cells < 4:
        raise ValueError("Use at least six groups and four cells per group")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "template").mkdir()
    (root / "holes").mkdir()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(groups)
    val_groups = set(perm[:max(1, groups // 6)])
    test_groups = set(perm[max(1, groups // 6):2 * max(1, groups // 6)])
    manifest = []
    for group in range(groups):
        split = "val" if group in val_groups else "test" if group in test_groups else "train"
        offset = rng.integers(-2, 3, size=2)
        for cell in range(cells):
            # Each group includes all labels; process variation is shared within each group.
            label = CLASSES[(cell + group) % len(CLASSES)]
            x, y = (32 + offset + rng.integers(-2, 3, size=2)).tolist()
            w, h = {
                "good": (18, 44), "line_or_bridge": (13, 46),
                "wrong_count": (19, 32), "other_defect": (24, 45),
            }[label]
            w += int(rng.integers(-2, 3))
            h += int(rng.integers(-3, 4))
            template = Image.new("L", (64, 64))
            ImageDraw.Draw(template).rounded_rectangle(
                (x-w//2, y-h//2, x+w//2, y+h//2), radius=w//2, fill=255)
            holes = Image.new("L", (64, 64))
            draw = ImageDraw.Draw(holes)
            radius = int(rng.integers(3, 5))
            centers = [y-12, y, y+12] if label != "wrong_count" else [y-7, y+7]
            for i, cy in enumerate(centers):
                if label == "other_defect" and i == 1:
                    draw.ellipse((x-7, cy-2, x+7, cy+2), fill=255)
                else:
                    dx = int(rng.integers(-1, 2))
                    draw.ellipse((x+dx-radius, cy-radius, x+dx+radius, cy+radius), fill=255)
            if label == "line_or_bridge":
                draw.line((x, centers[0], x, centers[-1]), fill=255, width=4)
            sample = f"g{group:03d}_c{cell:03d}"
            a = root / "template" / f"{sample}.png"
            b = root / "holes" / f"{sample}.png"
            template.save(a)
            holes.save(b)
            manifest.append(Sample(sample, f"g{group:03d}", a, b, 2.0, label, "synthetic", split))
    path = root / "manifest.csv"
    write_manifest(path, manifest)
    return path
