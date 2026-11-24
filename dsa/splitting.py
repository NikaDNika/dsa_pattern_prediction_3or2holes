"""Assign independent groups to dataset partitions."""

from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np

from .data import CLASSES, SPLITS


def group_split(samples, seed, val_fraction, test_fraction):
    if any(s.split for s in samples):
        raise ValueError("Use an entirely unsplit manifest; existing assignments are kept by training directly")
    grouped = defaultdict(Counter)
    for s in samples:
        grouped[s.group_id][s.label] += 1
    keys = sorted(grouped)
    n = len(keys)
    if n < 3 or val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Need at least 3 groups and positive fractions with val + test < 1")
    nv = max(1, round(n * val_fraction))
    nt = max(1, round(n * test_fraction))
    if nv + nt >= n:
        raise ValueError("Requested fractions leave no train groups; reduce val/test fractions")
    rows = np.array([[grouped[k][c] for c in CLASSES] for k in keys])
    total = rows.sum(axis=0)
    desired = np.array([1-val_fraction-test_fraction, val_fraction, test_fraction])
    rng = np.random.default_rng(seed)
    best, choice = None, None
    for _ in range(256):
        train, val, test = np.split(rng.permutation(n), [n - nv - nt, n - nt])
        counts = np.array([rows[indices].sum(axis=0) for indices in (train, val, test)])
        missing_train = int(np.sum((total > 0) & (counts[0] == 0)))
        missing_valid = int(np.sum((total > 0) & (counts[1:] == 0)))
        balance = abs(counts / np.maximum(total, 1) - desired[:, None]).sum()
        rank = (missing_train, missing_valid, balance)
        if best is None or rank < best:
            best, choice = rank, (train, val, test)
    assigned = {keys[i]: split for split, indices in zip(SPLITS, choice) for i in indices}
    return [replace(s, split=assigned[s.group_id]) for s in samples]
