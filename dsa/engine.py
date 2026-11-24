"""Training, reproducible validation, test reporting and prediction exports."""

import csv
import json
import math
import os
import random
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion
from torch.utils.data import DataLoader

from .data import CLASSES, Grid, MaskDataset, inputs_from_file, read_mask, to_native
from .metrics import classification, components, segmentation_scores
from dsa.models.unet import PatternUNet, multitask_loss
from dsa.preprocessing.segmentation import segment_file

VERSION = 1


def pick_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(name)


def json_file(path, values):
    path = Path(path)
    path.write_text(json.dumps(values, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _checkpoint(path, state):
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def load_checkpoint(path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    return _mask_model_from_state(state, device)


def _mask_model_from_state(state, device):
    if state.get("version") != VERSION or tuple(state.get("classes", ())) != CLASSES:
        raise ValueError("Unsupported checkpoint schema or label order")
    model = PatternUNet(state["width"], len(CLASSES)).to(device)
    model.load_state_dict(state["weights"])
    return model.eval(), Grid(**state["grid"]), state


def _loader(samples, grid, batch, shuffle=False, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(MaskDataset(samples, grid), batch_size=batch, shuffle=shuffle,
                      num_workers=0, generator=generator)


@torch.inference_mode()
def score(model, loader, device, grid, threshold, min_area, keep_rows=True):
    model.eval()
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    dices, ious, counts, three, rows = [], [], [], [], []
    for x, y, classes, ids in loader:
        masks, logits = model(x.to(device))
        probs = masks.sigmoid().cpu().numpy()[:, 0]
        scores = logits.softmax(dim=1).cpu().numpy()
        expected = y.numpy()[:, 0] > 0
        for sample, prob, cls_score, target, cls in zip(ids, probs, scores, expected, classes.tolist()):
            predicted = int(np.argmax(cls_score))
            confusion[cls, predicted] += 1
            binary = prob >= threshold
            dice, iou = segmentation_scores(binary, target)
            _, detected = components(binary, grid.nm_per_px, min_area)
            _, actual = components(target, grid.nm_per_px, min_area)
            dices.append(dice)
            ious.append(iou)
            counts.append(len(detected) == len(actual))
            three.append(len(detected) == 3)
            if keep_rows:
                rows.append({"sample_id": sample, "truth": CLASSES[cls],
                             "prediction": CLASSES[predicted], "dice": dice,
                             "predicted_components": len(detected), "target_components": len(actual),
                             **{f"score_{c}": float(s) for c, s in zip(CLASSES, cls_score)}})
    result = classification(confusion)
    result.update(
        pixel_dice=float(np.mean(dices)), pixel_iou=float(np.mean(ious)),
        component_count_accuracy=float(np.mean(counts)), predicted_three_fraction=float(np.mean(three)),
        sample_count=len(dices), mask_threshold=threshold, min_area_nm2=min_area,
    )
    return result, rows


def train(samples, grid, args):
    parts = {key: [s for s in samples if s.split == key] for key in ("train", "val", "test")}
    if any(not part for part in parts.values()):
        raise ValueError("Require nonempty train, val and test sets with separate groups")
    counts = Counter(s.label for s in parts["train"])
    if set(CLASSES) - counts.keys():
        raise ValueError("Train split must contain all classes")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    json_file(out / "config.json", {"arguments": config, "grid": asdict(grid),
                                    "device": str(device), "torch_version": str(torch.__version__)})
    train_batches = _loader(parts["train"], grid, args.batch, True, args.seed)
    validation = _loader(parts["val"], grid, args.batch)
    model = PatternUNet(args.width, len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    inv = np.array([1 / math.sqrt(counts[c]) for c in CLASSES])
    weights = torch.tensor(inv / inv.mean(), dtype=torch.float32, device=device)
    best, stale, history = -math.inf, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, size = 0.0, 0
        for x, y, c, _ in train_batches:
            x, y, c = x.to(device), y.to(device), c.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = multitask_loss(model(x), y, c, weights, args.class_weight)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.shape[0]
            size += x.shape[0]
        metrics, _ = score(model, validation, device, grid, args.mask_threshold,
                           args.min_area_nm2, keep_rows=False)
        # Penalize merged and fragmented openings as well as pixel/class errors.
        selection = (0.4 * metrics["pixel_dice"] + 0.3 * metrics["component_count_accuracy"]
                     + 0.3 * metrics["macro_f1_observed"])
        row = {"epoch": epoch, "train_loss": running / size, "selection": selection,
               "val": metrics}
        history.append(row)
        print(f"epoch={epoch} loss={running/size:.4f} dice={metrics['pixel_dice']:.4f} "
              f"count_acc={metrics['component_count_accuracy']:.4f} "
              f"macro_f1={metrics['macro_f1_observed']:.4f}", flush=True)
        if selection > best:
            best, stale = selection, 0
            _checkpoint(out / "best.pt", {
                "version": VERSION, "classes": list(CLASSES), "width": args.width,
                "grid": asdict(grid), "weights": model.state_dict(), "epoch": epoch,
                "data_kind": samples[0].data_kind, "majority_label": counts.most_common(1)[0][0],
                "groups": {key: sorted({s.group_id for s in part}) for key, part in parts.items()},
                "mask_threshold": args.mask_threshold, "min_area_nm2": args.min_area_nm2,
            })
        else:
            stale += 1
        json_file(out / "history.json", history)
        if stale >= args.patience:
            break
    return out / "best.pt"


def evaluate(samples, args):
    device = pick_device(args.device)
    model, grid, state = load_checkpoint(args.checkpoint, device)
    if samples[0].data_kind != state["data_kind"]:
        raise ValueError("Checkpoint and evaluation manifest have different data kinds")
    selected = [s for s in samples if s.split == args.split]
    if not selected:
        raise ValueError(f"No rows with split={args.split}")
    if args.split == "test" and set(s.group_id for s in selected) & set(state["groups"]["train"] + state["groups"]["val"]):
        raise ValueError("Test groups overlap training or validation groups in the checkpoint")
    result, rows = score(model, _loader(selected, grid, args.batch), device, grid,
                         state["mask_threshold"], state["min_area_nm2"])
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for s in selected:
        confusion[CLASSES.index(s.label), CLASSES.index(state["majority_label"])] += 1
    result.update(split=args.split, dataset_kind=state["data_kind"],
                  majority_baseline=classification(confusion))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    json_file(out / "metrics.json", result)
    with (out / "per_sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2), flush=True)


def _write_mask(path, mask):
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)


@torch.inference_mode()
def _predict_one(model, grid, state, device, template, spacing, out, review_score):
    x, native_template = inputs_from_file(template, spacing, grid)
    masks, class_logits = model(torch.from_numpy(x).unsqueeze(0).to(device))
    prob = masks.sigmoid()[0, 0].cpu().numpy()
    scores = class_logits.softmax(dim=1)[0].cpu().numpy()
    label_i = int(scores.argmax())
    confidence = float(scores[label_i])
    threshold, min_area = state["mask_threshold"], state["min_area_nm2"]
    clean, regions = components(prob >= threshold, grid.nm_per_px, min_area)
    native_prob = to_native(prob, native_template.shape, spacing, grid)
    native_mask, _ = components(native_prob >= threshold, spacing, min_area)
    review = []
    if confidence < review_score:
        review.append("low_class_score")
    if label_i == 0 and len(regions) != 3:
        review.append("good_class_but_component_count_is_not_three")
    if (prob >= threshold)[[0, -1], :].any() or (prob >= threshold)[:, [0, -1]].any():
        review.append("prediction_touches_canonical_edge")
    if state["data_kind"] == "synthetic":
        review.append("synthetic_checkpoint_for_demo_only")
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "hole_probability.npy", prob)
    np.save(out / "hole_probability_native.npy", native_prob)
    _write_mask(out / "predicted_holes.png", clean)
    _write_mask(out / "predicted_holes_native.png", native_mask)
    _write_mask(out / "template_canonical.png", x[0])
    image = np.full((*clean.shape, 3), (245, 246, 249), dtype=np.uint8)
    image[x[0] > 0] = (195, 209, 228)
    image[clean] = (196, 68, 51)
    Image.fromarray(image).save(out / "overlay.png")
    result = {
        "prediction": CLASSES[label_i],
        "class_scores_uncalibrated": {c: float(s) for c, s in zip(CLASSES, scores)},
        "defect_score_uncalibrated": 1 - float(scores[0]),
        "needs_review": bool(review), "review_reasons": review,
        "component_count": len(regions), "components_canonical": regions,
        "mask_threshold": threshold, "min_area_nm2": min_area, "grid": asdict(grid),
        "input_nm_per_px": spacing, "input_shape_yx": list(native_template.shape),
        "dataset_kind": state["data_kind"],
    }
    json_file(out / "prediction.json", result)
    return result


def predict(args):
    device = pick_device(args.device)
    model, grid, state = load_checkpoint(args.checkpoint, device)
    result = _predict_one(model, grid, state, device, args.template, args.nm_per_px,
                          args.out, args.review_score)
    print(f"{result['prediction']} | count={result['component_count']} | output={args.out}",
          flush=True)


def _write_input_overlay(source_path, roi_mask, segmentation, out, *, dark_mask=None, dark_only=False):
    left, top, width, height = segmentation["roi_xywh"]
    with Image.open(source_path) as source:
        if (roi_mask.shape != (height, width) or left < 0 or top < 0
                or left + width > source.width or top + height > source.height):
            raise ValueError("Predicted mask does not match the input ROI")
        if source.mode in ("I", "F") or source.mode.startswith("I;16"):
            # Display high-bit-depth SEM data using the segmentation ROI's contrast window.
            lo, hi = segmentation["intensity_p01"], segmentation["intensity_p99"]
            with np.errstate(invalid="ignore"):
                levels = np.clip((np.asarray(source, dtype=np.float32) - lo) / (hi - lo), 0, 1)
            gray = np.rint(np.nan_to_num(levels) * 255).astype(np.uint8)
            rgb = np.repeat(gray[..., None], 3, axis=2)
        else:
            rgb = np.array(source.convert("RGB"))
    Image.fromarray(rgb).save(out / "input_preview.png")
    full_mask = np.zeros(rgb.shape[:2], dtype=bool)
    region = np.s_[top:top + height, left:left + width]
    full_mask[region] = roi_mask
    _write_mask(out / "predicted_holes_on_input.png", full_mask)
    yellow = np.array([255, 220, 30], dtype=np.float32)
    if dark_mask is not None:
        if dark_mask.shape != roi_mask.shape:
            raise ValueError("Dark-region mask does not match the input ROI")
        full_dark = np.zeros_like(full_mask)
        full_dark[region] = dark_mask
        _write_mask(out / "predicted_dark_regions_on_input.png", full_dark)
        rgb[full_dark] = np.rint(.48 * rgb[full_dark] + .52 * yellow).astype(np.uint8)
        if not dark_only:
            iterations = max(1, round(1.5 / segmentation.get("nm_per_px", 1)))
            outline = roi_mask & ~binary_erosion(roi_mask, iterations=iterations)
            rgb[region][outline] = (255, 48, 32)
    else:
        color = yellow if dark_only else np.array([255, 48, 32], dtype=np.float32)
        rgb[full_mask] = np.rint(.48 * rgb[full_mask] + .52 * color).astype(np.uint8)
    Image.fromarray(rgb).save(out / "overlay_on_input.png")


def predict_image(args):
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    from dsa.training.reconstruction import MODEL_KIND, SIZE, SPACING, _model_from_state, predict_sem, rank_sem_openings

    device = pick_device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    is_sem = state.get("model_kind") == MODEL_KIND
    if is_sem:
        model = _model_from_state(state, device)
        grid = Grid(size=SIZE, nm_per_px=SPACING)
    else:
        model, grid, _ = _mask_model_from_state(state, device)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as scratch:
        stage = Path(scratch) / "image"
        stage.mkdir()
        segmentation = segment_file(args, stage / "segmentation", binary=args.binary_template)
        template = stage / "segmentation" / "template.png"
        dark_roi = None
        if is_sem:
            inputs, native_template = inputs_from_file(template, args.nm_per_px, grid)
            prediction_dir = stage / "prediction"
            result = predict_sem(model, state, device, inputs, prediction_dir, args.review_score)
            target_count = {"2孔": 2, "3孔": 3}[result["prediction"]]
            with Image.open(prediction_dir / "predicted_sem.png") as image:
                dark, dark_count, shape, baseline, candidates = rank_sem_openings(
                    np.asarray(image), inputs[0] > 0, target_count)
            _write_mask(prediction_dir / "predicted_dark_regions.png", dark)
            _write_mask(prediction_dir / "predicted_hole_shapes.png", shape)
            dark_roi = (to_native(dark.astype(np.float32), native_template.shape,
                                  args.nm_per_px, grid) >= .5) & native_template
            roi_mask = (to_native(shape.astype(np.float32), native_template.shape,
                                  args.nm_per_px, grid) >= .5) & dark_roi
            roi_mask, drawn_regions = components(roi_mask, args.nm_per_px, min_area_nm2=20)
            result.update(dark_region_count=dark_count, hole_shape_available=True,
                          hole_shape_count=len(drawn_regions), target_hole_count=target_count,
                          prediction_source="classifier", hole_selection_method="ranked_sem_darkness",
                          cavity_reference_gray=baseline, dark_candidates=candidates)
            if sum(c["selected"] for c in candidates) < target_count:
                result["review_reasons"].append("sem_dark_candidates_shortfall")
            elif len(drawn_regions) != target_count:
                result["review_reasons"].append("sem_selected_count_changed_on_native_grid")
            json_file(prediction_dir / "prediction.json", result)
        else:
            result = _predict_one(model, grid, state, device, template, args.nm_per_px,
                                  stage / "prediction", args.review_score)
            roi_mask = read_mask(stage / "prediction" / "predicted_holes_native.png")
        _write_input_overlay(args.image, roi_mask, segmentation, stage, dark_mask=dark_roi,
                             dark_only=args.dark_only_overlay)
        stage.rename(out)
    print(f"{result['prediction']} | output={out}", flush=True)


def _prediction_rows(manifest):
    manifest = manifest.resolve()
    seen = set()
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"sample_id", "template_path", "nm_per_px"}.issubset(reader.fieldnames or ()):
            raise ValueError("Batch CSV requires sample_id, template_path and nm_per_px")
        for row in reader:
            index = reader.line_num
            sample = (row["sample_id"] or "").strip()
            path = (manifest.parent / (row["template_path"] or "").strip()).resolve()
            try:
                spacing = float(row["nm_per_px"] or "")
            except ValueError as exc:
                raise ValueError(f"Row {index}: invalid nm_per_px") from exc
            if not sample or sample in seen or not path.is_file() or not math.isfinite(spacing) or spacing <= 0:
                raise ValueError(f"Row {index}: invalid sample_id, template file or nm_per_px")
            seen.add(sample)
            yield sample, path, spacing


def predict_batch(args):
    inputs = list(_prediction_rows(args.manifest))
    if not inputs:
        raise ValueError("Batch manifest is empty")
    device = pick_device(args.device)
    model, grid, state = load_checkpoint(args.checkpoint, device)
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as scratch:
        stage = Path(scratch) / "batch"
        stage.mkdir()
        rows = []
        for i, (sample, path, spacing) in enumerate(inputs, 1):
            folder = f"cell_{i:05d}"
            try:
                result = _predict_one(model, grid, state, device, path, spacing,
                                      stage / folder, args.review_score)
            except ValueError as exc:
                raise ValueError(f"{sample}: {exc}") from exc
            rows.append({"sample_id": sample, "prediction": result["prediction"],
                         "defect_score_uncalibrated": result["defect_score_uncalibrated"],
                         "component_count": result["component_count"], "needs_review": result["needs_review"],
                         "review_reasons": "; ".join(result["review_reasons"]), "output_folder": folder})
        rows.sort(key=lambda item: -item["defect_score_uncalibrated"])
        with (stage / "ranking.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        stage.rename(out)
    print(f"Scored {len(rows)} templates | output={out}", flush=True)
