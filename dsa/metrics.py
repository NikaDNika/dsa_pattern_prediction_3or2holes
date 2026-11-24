"""Segmentation, defect classification and connected-hole summaries."""

import math

import numpy as np
from scipy.ndimage import label as label_image

from .data import CLASSES


def components(mask, spacing, min_area_nm2):
    indices, n = label_image(mask, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(indices.ravel(), minlength=n + 1)
    valid = np.flatnonzero((areas * spacing**2 >= min_area_nm2) & (areas > 0))
    valid = valid[valid != 0]
    clean = np.isin(indices, valid)
    stats = []
    x0, y0 = (mask.shape[1] - 1) / 2, (mask.shape[0] - 1) / 2
    for index in valid:
        y, x = np.nonzero(indices == index)
        area = len(x) * spacing**2
        eig = np.linalg.eigvalsh(np.cov(np.stack([x, y]), bias=True)) if len(x) > 1 else [0, 0]
        stats.append({
            "center_x_nm": float((x.mean() - x0) * spacing),
            "center_y_nm": float((y.mean() - y0) * spacing),
            "area_nm2": area,
            "equivalent_diameter_nm": 2 * math.sqrt(area / math.pi),
            "elongation": float(math.sqrt(eig[1] / eig[0])) if eig[0] > 0 else None,
        })
    return clean, sorted(stats, key=lambda s: (s["center_y_nm"], s["center_x_nm"]))


def classification(confusion):
    n = int(confusion.sum())
    if not n:
        raise ValueError("No samples to score")
    result = {}
    for i, name in enumerate(CLASSES):
        tp = int(confusion[i, i])
        support = int(confusion[i, :].sum())
        predictions = int(confusion[:, i].sum())
        result[name] = {
            "support": support,
            "precision": tp / predictions if predictions else None,
            "recall": tp / support if support else None,
            "f1": 2 * tp / (support + predictions) if support else None,
        }
    observed = [item["f1"] for item in result.values() if item["support"]]
    defect_true = int(confusion[1:, :].sum())
    defect_tp = int(confusion[1:, 1:].sum())
    good_true = int(confusion[0, :].sum())
    pred_defects = int(confusion[:, 1:].sum())
    return {
        "accuracy": float(np.trace(confusion) / n),
        "macro_f1_observed": sum(observed) / len(observed),
        "macro_f1_all_classes": sum(observed) / len(CLASSES) if len(observed) == len(CLASSES) else None,
        "class_metrics": result,
        "confusion_matrix": confusion.tolist(),
        "class_order": list(CLASSES),
        "defect_recall": defect_tp / defect_true if defect_true else None,
        "defect_precision": defect_tp / pred_defects if pred_defects else None,
        "false_pass_rate_among_true_defects": float(confusion[1:, 0].sum() / defect_true) if defect_true else None,
        "false_reject_rate_among_true_good": float(confusion[0, 1:].sum() / good_true) if good_true else None,
    }


def segmentation_scores(prediction, truth):
    intersect = np.count_nonzero(prediction & truth)
    total = np.count_nonzero(prediction) + np.count_nonzero(truth)
    union = total - intersect
    return (2 * intersect / total if total else 1, intersect / union if union else 1)
