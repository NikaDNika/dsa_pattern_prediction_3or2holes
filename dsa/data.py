"""Manifest checks and scale-preserving preprocessing for registered binary masks."""

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt, map_coordinates
from torch.utils.data import Dataset

CLASSES = ("good", "line_or_bridge", "wrong_count", "other_defect")
SPLITS = ("train", "val", "test")
KINDS = ("synthetic", "experimental")
FIELDS = ("sample_id", "group_id", "template_path", "holes_path",
          "nm_per_px", "label", "data_kind", "split")


@dataclass(frozen=True)
class Grid:
    size: int = 256
    nm_per_px: float = 0.78125
    sdf_clip_nm: float = 24.0

    def __post_init__(self):
        if self.size < 32 or self.size % 8:
            raise ValueError("size must be a multiple of 8 and at least 32")
        for value in (self.nm_per_px, self.sdf_clip_nm):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("pixel spacing and SDF clipping distance must be positive")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    group_id: str
    template_path: Path
    holes_path: Path
    nm_per_px: float
    label: str
    data_kind: str
    split: str


def load_manifest(path, require_split=True):
    path = Path(path).resolve()
    samples = []
    sample_ids, input_paths, target_paths = set(), set(), set()
    group_splits = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(FIELDS if require_split else FIELDS[:-1])
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Required CSV columns: {', '.join(sorted(required))}")
        for row in reader:
            number = reader.line_num
            row = {k: (row.get(k) or "").strip() for k in FIELDS}
            sample_id, group_id = row["sample_id"], row["group_id"]
            if not sample_id or not group_id or sample_id in sample_ids:
                raise ValueError(f"Row {number}: empty identifier or duplicate sample_id")
            try:
                spacing = float(row["nm_per_px"])
            except ValueError as exc:
                raise ValueError(f"Row {number}: invalid nm_per_px") from exc
            if not math.isfinite(spacing) or spacing <= 0:
                raise ValueError(f"Row {number}: nm_per_px must be positive")
            if row["label"] not in CLASSES or row["data_kind"] not in KINDS:
                raise ValueError(f"Row {number}: unknown label or data_kind")
            split = row["split"]
            if (require_split or split) and split not in SPLITS:
                raise ValueError(f"Row {number}: invalid split")
            if split and group_splits.setdefault(group_id, split) != split:
                raise ValueError(f"Row {number}: group {group_id} spans multiple splits")
            paths = [(path.parent / row[k]).resolve() for k in ("template_path", "holes_path")]
            if not all(p.is_file() for p in paths):
                raise ValueError(f"Row {number}: missing mask file")
            if paths[0] == paths[1] or paths[0] in input_paths or paths[1] in target_paths:
                raise ValueError(f"Row {number}: repeated mask path; use one independent row per cell")
            if paths[0] in target_paths or paths[1] in input_paths:
                raise ValueError(f"Row {number}: a target file was reused as a template, or vice versa")
            input_paths.add(paths[0])
            target_paths.add(paths[1])
            sample_ids.add(sample_id)
            samples.append(Sample(sample_id, group_id, *paths, spacing,
                                  row["label"], row["data_kind"], split))
    if not samples:
        raise ValueError("Manifest is empty")
    if len({sample.data_kind for sample in samples}) != 1:
        raise ValueError("Use separate manifests for experimental and synthetic data")
    return samples


def write_manifest(path, samples):
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for sample in samples:
            row = asdict(sample)
            for key in ("template_path", "holes_path"):
                try:
                    row[key] = row[key].resolve().relative_to(path.parent.resolve()).as_posix()
                except ValueError:
                    row[key] = row[key].resolve().as_posix()
            writer.writerow(row)


def read_mask(path):
    # RGB or greyscale SEM input requires explicit segmentation first.
    with Image.open(path) as image:
        if image.mode not in ("1", "L"):
            raise ValueError(f"{path}: expected a single-channel binary mask, mode 1 or L")
        pixels = np.asarray(image)
    if not (np.isin(pixels, [0, 1]).all() or np.isin(pixels, [0, 255]).all()):
        raise ValueError(f"{path}: expected binary pixel values 0/1 or 0/255")
    return pixels > 0


def _resample(array, size, scale, order):
    # Each array is centered at ((W - 1)/2, (H - 1)/2), preserving X/Y scale.
    h, w = array.shape
    yy = (np.arange(size[0]) - (size[0] - 1) / 2) * scale + (h - 1) / 2
    xx = (np.arange(size[1]) - (size[1] - 1) / 2) * scale + (w - 1) / 2
    coords = np.meshgrid(yy, xx, indexing="ij")
    return map_coordinates(array, coords, order=order, mode="grid-constant", cval=0)


def _ensure_complete(binary, name, empty_ok=False):
    if not empty_ok and not binary.any():
        raise ValueError(f"{name} is empty")
    if binary[[0, -1], :].any() or binary[:, [0, -1]].any():
        raise ValueError(f"{name} touches the crop edge; use a larger crop or grid")


def _registered_mask(binary, spacing, grid, name, empty_ok=False):
    _ensure_complete(binary, name, empty_ok)
    ys, xs = np.nonzero(binary)
    if len(ys):
        y_extent = np.max(np.abs(ys - (binary.shape[0] - 1) / 2))
        x_extent = np.max(np.abs(xs - (binary.shape[1] - 1) / 2))
        required_span = 2 * (max(y_extent, x_extent) + 0.5) * spacing
        usable_span = (grid.size - 1) * grid.nm_per_px
        if required_span >= usable_span:
            raise ValueError(
                f"{name} exceeds the canonical field of view: input {binary.shape[1]}x{binary.shape[0]} px "
                f"at {spacing:g} nm/px requires a centered span of {required_span:.1f} nm; "
                f"checkpoint usable span is {usable_span:.1f} nm. "
                "Check the source dimensions and pixel calibration, or use a checkpoint with a larger grid."
            )
    resized = _resample(binary.astype(np.uint8), (grid.size, grid.size),
                        grid.nm_per_px / spacing, order=0).astype(bool)
    _ensure_complete(resized, f"resampled {name}", empty_ok)
    if binary.any() and not resized.any():
        raise ValueError(f"{name} disappears after resampling; use a finer grid")
    return resized


def inputs_from_file(path, spacing, grid):
    original = read_mask(path)
    template = _registered_mask(original, spacing, grid, "template")
    sdf = (distance_transform_edt(template) - distance_transform_edt(~template))
    sdf *= grid.nm_per_px / grid.sdf_clip_nm
    tensor = np.stack([template, np.clip(sdf, -1, 1)]).astype(np.float32)
    return tensor, original


def to_native(probabilities, original_shape, spacing, grid):
    return _resample(probabilities, original_shape, spacing / grid.nm_per_px, order=1)


class MaskDataset(Dataset):
    def __init__(self, samples, grid):
        self.samples = list(samples)
        self.grid = grid

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        try:
            x, original = inputs_from_file(item.template_path, item.nm_per_px, self.grid)
            y = read_mask(item.holes_path)
            if y.shape != original.shape:
                raise ValueError("input and target are on different pixel grids")
            y = _registered_mask(y, item.nm_per_px, self.grid, "holes", empty_ok=True)
        except ValueError as exc:
            raise ValueError(f"{item.sample_id}: {exc}") from exc
        return (torch.from_numpy(x), torch.from_numpy(y[None].astype(np.float32)),
                CLASSES.index(item.label), item.sample_id)
