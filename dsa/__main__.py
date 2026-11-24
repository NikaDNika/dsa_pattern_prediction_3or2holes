"""Command line for the DSA three-hole prediction project."""

import argparse
from collections import Counter
import math
from pathlib import Path

import torch

from .data import CLASSES, SPLITS, Grid, MaskDataset, load_manifest, write_manifest
from .engine import evaluate, predict, predict_batch, predict_image, train
from dsa.preprocessing.segmentation import run_segmentation
from .splitting import group_split
from .synthetic import make_demo


def positive_float(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return value


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def probability(text):
    value = float(text)
    if not 0 < value < 1:
        raise argparse.ArgumentTypeError("expected a number in (0, 1)")
    return value


def grid_args(parser):
    defaults = Grid()
    parser.add_argument("--size", type=positive_int, default=defaults.size)
    parser.add_argument("--grid-nm-per-px", type=positive_float, default=defaults.nm_per_px)
    parser.add_argument("--sdf-clip-nm", type=positive_float, default=24.0)


def compute_args(parser):
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=positive_int, default=4)


def image_args(command, *, require_spacing=False):
    command.add_argument("--image", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--roi", nargs=4, type=int, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"))
    command.add_argument("--polarity", choices=("auto", "dark", "bright"), default="auto")
    command.add_argument("--sigma", type=float, default=1.0)
    command.add_argument("--threshold", type=probability, help="override Otsu with a normalized threshold")
    command.add_argument("--nm-per-px", type=positive_float, required=require_spacing,
                         help="existing physical pixel scale of the input")


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo-data", help="generate a synthetic pipeline fixture")
    demo.add_argument("--out", type=Path, required=True)
    demo.add_argument("--groups", type=positive_int, default=18)
    demo.add_argument("--cells-per-group", type=positive_int, default=12)
    demo.add_argument("--seed", type=int, default=42)

    split = commands.add_parser("split", help="assign train/val/test by group")
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--out", type=Path, required=True)
    split.add_argument("--val-fraction", type=probability, default=0.2)
    split.add_argument("--test-fraction", type=probability, default=0.2)
    split.add_argument("--seed", type=int, default=42)

    verify = commands.add_parser("validate", help="check masks, scale, and manifest partitions")
    verify.add_argument("--manifest", type=Path, required=True)
    grid_args(verify)

    fit = commands.add_parser("train", help="fit U-Net and select the best validation checkpoint")
    fit.add_argument("--manifest", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--epochs", type=positive_int, default=40)
    fit.add_argument("--batch", type=positive_int, default=8)
    fit.add_argument("--width", type=positive_int, default=16)
    fit.add_argument("--patience", type=positive_int, default=8)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--lr", type=positive_float, default=0.001)
    fit.add_argument("--class-weight", type=positive_float, default=0.5)
    fit.add_argument("--mask-threshold", type=probability, default=0.5)
    fit.add_argument("--min-area-nm2", type=positive_float, default=16.0)
    grid_args(fit)
    compute_args(fit)

    test = commands.add_parser("evaluate", help="score a locked dataset partition")
    test.add_argument("--manifest", type=Path, required=True)
    test.add_argument("--checkpoint", type=Path, required=True)
    test.add_argument("--out", type=Path, required=True)
    test.add_argument("--split", choices=("val", "test"), default="test")
    test.add_argument("--batch", type=positive_int, default=8)
    compute_args(test)

    seg = commands.add_parser("segment-template", help="binarize one registered pre-DSA template image")
    image_args(seg)

    raw = commands.add_parser("predict-image", help="segment and predict from one registered pre-DSA image")
    image_args(raw, require_spacing=True)
    raw.add_argument("--checkpoint", type=Path, required=True)
    raw.add_argument("--binary-template", action="store_true",
                     help="use the supplied white cavity mask directly, without thresholding")
    raw.add_argument("--dark-only-overlay", action="store_true",
                     help="display a yellow-only overlay while retaining both result masks")
    raw.add_argument("--review-score", type=probability, default=0.7)
    compute_args(raw)

    infer = commands.add_parser("predict", help="predict one pre-DSA template")
    infer.add_argument("--template", type=Path, required=True)
    infer.add_argument("--nm-per-px", type=positive_float, required=True)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--out", type=Path, required=True)
    infer.add_argument("--review-score", type=probability, default=0.7)
    compute_args(infer)
    batch_infer = commands.add_parser("predict-batch", help="rank templates by defect score")
    batch_infer.add_argument("--manifest", type=Path, required=True)
    batch_infer.add_argument("--checkpoint", type=Path, required=True)
    batch_infer.add_argument("--out", type=Path, required=True)
    batch_infer.add_argument("--review-score", type=probability, default=0.7)
    compute_args(batch_infer)
    return root


def describe(samples):
    for split in SPLITS:
        part = [s for s in samples if s.split == split]
        counts = Counter(s.label for s in part)
        print(f"{split}: {len(part)} samples, {len({s.group_id for s in part})} groups, "
              f"{dict((label, counts[label]) for label in CLASSES)}")


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        if hasattr(args, "threads"):
            torch.set_num_threads(args.threads)
        if args.command == "demo-data":
            print(make_demo(args.out, args.groups, args.cells_per_group, args.seed))
        elif args.command == "split":
            samples = load_manifest(args.manifest, require_split=False)
            samples = group_split(samples, args.seed, args.val_fraction, args.test_fraction)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            write_manifest(args.out, samples)
            describe(samples)
        elif args.command == "segment-template":
            run_segmentation(args)
        elif args.command == "predict-image":
            predict_image(args)
        elif args.command == "predict":
            predict(args)
        elif args.command == "predict-batch":
            predict_batch(args)
        else:
            samples = load_manifest(args.manifest)
            if args.command == "evaluate":
                evaluate(samples, args)
            else:
                grid = Grid(args.size, args.grid_nm_per_px, args.sdf_clip_nm)
                describe(samples)
                if args.command == "validate":
                    dataset = MaskDataset(samples, grid)
                    for index in range(len(dataset)):
                        dataset[index]
                    print(f"Validated {len(samples)} registered mask pairs")
                else:
                    print(f"Best checkpoint: {train(samples, grid, args)}")
    except (ValueError, OSError) as exc:
        cli.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
