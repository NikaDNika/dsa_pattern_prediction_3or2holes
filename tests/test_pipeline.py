import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dsa.data import CLASSES, Grid, MaskDataset, inputs_from_file, load_manifest, to_native, write_manifest
from dsa.metrics import classification, components, segmentation_scores
from dsa.models.unet import PatternUNet, multitask_loss
from dsa.splitting import group_split
from dsa.synthetic import make_demo


class PipelineTests(unittest.TestCase):
    def test_grid_round_trip_and_sdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mask.png"
            arr = np.zeros((64, 64), np.uint8)
            arr[12:53, 24:39] = 255
            Image.fromarray(arr).save(path)
            grid = Grid(128, 1, 16)
            x, source = inputs_from_file(path, 2, grid)
            self.assertGreater(x[1, 64, 64], 0)
            self.assertLess(x[1, 0, 0], 0)
            reconstructed = to_native(x[0], source.shape, 2, grid) > 0.5
            np.testing.assert_array_equal(source, reconstructed)
            with self.assertRaisesRegex(ValueError, "field of view"):
                inputs_from_file(path, 2, Grid(64, 1, 16))

    def test_manifest_and_group_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_demo(Path(tmp) / "data", 6, 4, 7)
            manifest = load_manifest(path)
            self.assertEqual(len(manifest), 24)
            x, y, _, _ = MaskDataset(manifest, Grid(64, 2))[0]
            self.assertEqual(tuple(x.shape), (2, 64, 64))
            self.assertEqual(tuple(y.shape), (1, 64, 64))
            unsplit = [replace(s, split="") for s in manifest]
            assigned = group_split(unsplit, 7, 0.2, 0.2)
            self.assertEqual(assigned, group_split(unsplit, 7, 0.2, 0.2))
            self.assertEqual({s.split for s in assigned}, {"train", "val", "test"})
            self.assertTrue(all(len({s.split for s in assigned if s.group_id == g}) == 1
                                for g in {s.group_id for s in assigned}))
            bad = Path(tmp) / "leaked.csv"
            broken = list(manifest)
            broken[0] = replace(broken[0], split="test" if broken[0].split != "test" else "val")
            write_manifest(bad, broken)
            with self.assertRaisesRegex(ValueError, "spans multiple splits"):
                load_manifest(bad)

    def test_topology_and_missing_classes(self):
        y = np.zeros((32, 32), dtype=bool)
        for x in (4, 12, 20):
            y[5:10, x:x+4] = True
        y[25, 25] = True
        clean, stats = components(y, spacing=2, min_area_nm2=8)
        self.assertEqual(len(stats), 3)
        self.assertFalse(clean[25, 25])
        self.assertEqual(segmentation_scores(y, y), (1, 1))
        y[7, 7:21] = True
        self.assertEqual(len(components(y, 2, 8)[1]), 1)
        confusion = np.zeros((len(CLASSES), len(CLASSES)), int)
        confusion[0, 0] = 3
        self.assertIsNone(classification(confusion)["macro_f1_all_classes"])

    def test_model_backward(self):
        torch.set_num_threads(2)
        model = PatternUNet(4, len(CLASSES))
        x = torch.randn(2, 2, 64, 64)
        out = model(x)
        self.assertEqual(tuple(out[0].shape), (2, 1, 64, 64))
        self.assertEqual(tuple(out[1].shape), (2, len(CLASSES)))
        loss = multitask_loss(out, torch.zeros_like(out[0]), torch.tensor([0, 1]), None, 1.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
