"""Extract one template cavity from a registered pre-DSA intensity image."""

import json

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, binary_fill_holes, gaussian_filter, label

CONNECTIVITY = np.ones((3, 3), dtype=bool)


def _load_image(path, roi):
    with Image.open(path) as source:
        bounds = (0, 0, source.width, source.height)
        if roi is not None:
            x, y, w, h = roi
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > source.width or y + h > source.height:
                raise ValueError("ROI must be fully inside the input image")
            bounds = (x, y, x + w, y + h)
        image = source.crop(bounds)
        # Preserve intensity precision for 16-bit TIFF/PNG and float TIFF.
        if image.mode not in ("1", "L", "I", "F") and not image.mode.startswith("I;16"):
            image = image.convert("L")
        pixels = np.asarray(image, dtype=np.float32)
    return pixels, (bounds[0], bounds[1], pixels.shape[1], pixels.shape[0])


def _otsu(image):
    count, edges = np.histogram(image, bins=256, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = count.cumsum()[:-1]
    w1 = count.sum() - w0
    moments = (count * centers).cumsum()
    valid = (w0 > 0) & (w1 > 0)
    if not valid.any():
        raise ValueError("No usable intensity separation")
    scores = np.full(w0.shape, -np.inf)
    mean0 = moments[:-1][valid] / w0[valid]
    mean1 = (moments[-1] - moments[:-1][valid]) / w1[valid]
    scores[valid] = w0[valid] * w1[valid] * (mean0 - mean1)**2
    ties = np.flatnonzero(np.isclose(scores, scores.max(), rtol=1e-10, atol=1e-12))
    return float((edges[ties[0] + 1] + edges[ties[-1] + 1]) / 2)


def segment(pixels, polarity="auto", sigma=1.0, threshold=None):
    if pixels.ndim != 2 or min(pixels.shape) < 16 or not np.isfinite(pixels).all():
        raise ValueError("Need a finite single-cell grayscale crop at least 16 x 16 pixels")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Smoothing sigma must be finite and nonnegative")
    if polarity not in ("auto", "dark", "bright"):
        raise ValueError("Polarity must be auto, dark, or bright")
    if threshold is not None and not 0 < threshold < 1:
        raise ValueError("Threshold must be in (0, 1)")
    lo, hi = np.percentile(pixels, [1, 99])
    if hi <= lo:
        raise ValueError("Image has no usable intensity contrast between the 1st and 99th percentiles")
    normalized = np.clip((pixels - lo) / (hi - lo), 0, 1)
    filtered = gaussian_filter(normalized, sigma) if sigma else normalized
    if polarity == "auto":
        h, w = filtered.shape
        strip = max(1, min(h, w) // 16)
        border = np.concatenate((filtered[:strip].ravel(), filtered[-strip:].ravel(),
                                 filtered[strip:-strip, :strip].ravel(), filtered[strip:-strip, -strip:].ravel()))
        center = filtered[3*h//8:5*h//8, 3*w//8:5*w//8]
        contrast = float(np.median(center) - np.median(border))
        if abs(contrast) < 0.04:
            raise ValueError("Cannot infer polarity from center vs border; use --polarity dark or bright")
        polarity = "dark" if contrast < 0 else "bright"
    cut = _otsu(filtered) if threshold is None else threshold
    candidate = filtered <= cut if polarity == "dark" else filtered >= cut
    labels, n = label(candidate, structure=CONNECTIVITY)
    if not n:
        raise ValueError("No foreground found; check polarity and threshold")
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    mask = labels == counts.argmax()
    if mask[[0, -1], :].any() or mask[:, [0, -1]].any():
        raise ValueError("Largest foreground touches the crop edge; check polarity, threshold, or crop margin")
    # A template mask represents the whole cavity, including its interior.
    mask = binary_fill_holes(mask)
    y, x = np.nonzero(mask)
    if mask.sum() < 16:
        raise ValueError("Foreground is too small to represent a template; check threshold and crop")
    return mask, normalized, {
        "polarity": polarity,
        "threshold_normalized": cut,
        "threshold_method": "otsu" if threshold is None else "manual",
        "smooth_sigma_px": sigma,
        "intensity_p01": float(lo),
        "intensity_p99": float(hi),
        "foreground_pixels": int(mask.sum()),
        "foreground_bbox_xywh": [int(x.min()), int(y.min()), int(x.max()-x.min()+1), int(y.max()-y.min()+1)],
    }


def segment_file(args, out, *, binary=False):
    pixels, roi = _load_image(args.image, args.roi)
    if binary:
        levels = np.unique(pixels)
        if tuple(levels) not in ((0, 1), (0, 255)):
            raise ValueError("Binary template must have black background and a white cavity (0/255 or 0/1)")
        mask = pixels > 0
        if label(mask, structure=CONNECTIVITY)[1] != 1 or not np.array_equal(mask, binary_fill_holes(mask)):
            raise ValueError("Binary template must contain one filled cavity")
        y, x = np.nonzero(mask)
        gray = mask
        report = {"polarity": "bright", "threshold_method": "provided_mask",
                  "intensity_p01": 0.0, "intensity_p99": float(levels[-1]),
                  "foreground_pixels": int(mask.sum()),
                  "foreground_bbox_xywh": [int(x.min()), int(y.min()),
                                           int(x.max()-x.min()+1), int(y.max()-y.min()+1)]}
    else:
        mask, gray, report = segment(pixels, args.polarity, args.sigma, args.threshold)
    if args.nm_per_px is not None:
        report["nm_per_px"] = args.nm_per_px
        report["foreground_area_nm2"] = report["foreground_pixels"] * args.nm_per_px**2
    report.update(source=str(args.image.resolve()), roi_xywh=list(roi),
                  output_shape_yx=list(mask.shape), pixel_grid_resampled=False)
    gray = np.rint(gray * 255).astype(np.uint8)
    overlay = np.repeat(gray[..., None], 3, axis=2)
    outline = mask & ~binary_erosion(mask, structure=CONNECTIVITY)
    overlay[outline] = (255, 65, 48)
    out.mkdir(parents=True, exist_ok=False)
    Image.fromarray(mask.astype(np.uint8) * 255).save(out / "template.png")
    Image.fromarray(overlay).save(out / "overlay.png")
    (out / "segmentation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def run_segmentation(args):
    report = segment_file(args, args.out)
    print(f"{args.out / 'template.png'} | {report['polarity']} | foreground={report['foreground_pixels']} px")
