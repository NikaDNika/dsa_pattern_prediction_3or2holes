import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from dsa.data import read_mask
from dsa.engine import _write_input_overlay
from dsa.training.pore_shapes import trace_sem_pores
from dsa.training.reconstruction import rank_sem_openings, sem_preview_mask
from dsa.preprocessing.segmentation import run_segmentation, segment


class SegmentationTests(unittest.TestCase):
    def test_both_polarities_with_noise(self):
        y, x = np.indices((80, 64))
        expected = (x - 32)**2 / 9**2 + (y - 40)**2 / 24**2 <= 1
        noise = np.random.default_rng(17).normal(0, 0.03, expected.shape)
        for polarity, signal in [("bright", expected), ("dark", ~expected)]:
            with self.subTest(polarity=polarity):
                image = 0.25 + 0.5 * signal + 0.001 * x + noise
                mask, _, report = segment(image)
                iou = (mask & expected).sum() / (mask | expected).sum()
                self.assertGreater(iou, 0.96)
                self.assertEqual(report["polarity"], polarity)

    def test_ambiguous_or_clipped_images(self):
        with self.assertRaisesRegex(ValueError, "intensity contrast"):
            segment(np.full((64, 64), 128))
        clipped = np.ones((64, 64))
        clipped[:45, 21:41] = 0
        with self.assertRaisesRegex(ValueError, "touches the crop edge"):
            segment(clipped, "dark")

    def test_hole_overlay_uses_full_original_coordinates(self):
        original = np.full((30, 48, 3), (70, 90, 110), dtype=np.uint8)
        holes = np.zeros((12, 16), dtype=bool)
        holes[3:6, 7:9] = True
        expected = np.zeros(original.shape[:2], dtype=bool)
        expected[8:20, 15:31] = holes
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source = out / "source.png"
            Image.fromarray(original).save(source)
            _write_input_overlay(source, holes, {"roi_xywh": [15, 8, 16, 12]}, out)
            np.testing.assert_array_equal(read_mask(out / "predicted_holes_on_input.png"), expected)
            with Image.open(out / "overlay_on_input.png") as preview:
                changed = np.any(np.asarray(preview) != original, axis=2)
            np.testing.assert_array_equal(changed, expected)

    def test_sem_overlay_uses_predicted_dark_regions(self):
        y, x = np.indices((256, 256))
        template = ((x-128)/42)**2 + ((y-128)/99)**2 < 1
        base = np.where(template, 100, 190).astype(np.uint8)
        mask, count = sem_preview_mask(base, template)
        self.assertFalse(mask.any())
        self.assertEqual(count, 0)
        for components, centers in ((2, (88, 168)), (3, (82, 128, 174))):
            predicted = base.copy()
            for cy in centers:
                predicted[(x-128)**2 + (y-cy)**2 < 12**2] = 48
            mask, count = sem_preview_mask(predicted, template)
            self.assertEqual(count, components)
            self.assertTrue(all(mask[cy,128] for cy in centers))
            self.assertFalse(np.any(mask[~template]))

    def test_dual_overlay_distinguishes_dark_fill_from_pore_outline(self):
        original = np.full((24, 24, 3), 80, dtype=np.uint8)
        red = np.zeros((12,12),dtype=bool)
        red[4:9,4:9] = True
        yellow = np.zeros_like(red)
        yellow[2:10,2:10] = True
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            source = out/"source.png"
            Image.fromarray(original).save(source)
            _write_input_overlay(source,red,{"roi_xywh":[4,6,12,12]},out,dark_mask=yellow)
            with Image.open(out/"overlay_on_input.png") as image:
                result = np.asarray(image)
            self.assertEqual(tuple(result[10,8]),(255,48,32))
            self.assertGreater(result[9,7,1],result[9,7,2])
            self.assertGreater(result[12,10,1],result[12,10,2])
            self.assertEqual(tuple(result[0,0]),(80,80,80))
            self.assertEqual(int(read_mask(out/"predicted_dark_regions_on_input.png").sum()),64)
            self.assertEqual(int(read_mask(out/"predicted_holes_on_input.png").sum()),25)
            _write_input_overlay(source,red,{"roi_xywh":[4,6,12,12]},out,
                                 dark_mask=yellow,dark_only=True)
            with Image.open(out/"overlay_on_input.png") as image:
                preview = np.asarray(image)
            self.assertFalse(np.any(np.all(preview==(255,48,32),axis=-1)))
            self.assertEqual(tuple(preview[10,8]),tuple(preview[9,7]))
            self.assertEqual(int(read_mask(out/"predicted_holes_on_input.png").sum()),25)

    def test_edge_labels_follow_elongated_pores(self):
        y,x=np.indices((256,256))
        cavity=((x-128)/44)**2+((y-128)/100)**2<1
        spots=((x-128)/13)**2+((y-84)/17)**2<1
        spots|=((x-128)/10)**2+((y-168)/15)**2<1
        sem=np.where(cavity,111,205).astype(np.uint8)
        sem[spots]=54
        result=trace_sem_pores(sem,cavity,2)
        self.assertIsNotNone(result)
        self.assertGreater((result & spots).sum()/(result | spots).sum(),.72)
        self.assertIsNone(trace_sem_pores(sem,cavity,3))

    def test_sem_selection_prefers_deeper_regions_and_does_not_fill_shortfall(self):
        y, x = np.indices((256,256))
        template = ((x-128)/43)**2 + ((y-128)/104)**2 < 1
        predicted = np.where(template,112,195).astype(np.uint8)
        for cy, value in ((78,37),(126,69),(177,48)):
            predicted[(x-128)**2+(y-cy)**2 <= 13**2] = value
        dark, count, chosen, _, data = rank_sem_openings(predicted,template,2)
        self.assertEqual(count,3)
        self.assertEqual([entry["center_xy_px"][1] for entry in data], [78,177,126])
        self.assertEqual([entry["selected"] for entry in data],[True,True,False])
        self.assertTrue(chosen[78,128] and chosen[177,128])
        self.assertFalse(chosen[126,128])
        self.assertFalse(np.any(chosen & ~dark))

        predicted[(x-128)**2+(y-126)**2 <= 13**2] = 112
        _, _, _, _, data = rank_sem_openings(predicted,template,3)
        self.assertEqual(sum(entry["selected"] for entry in data),2)

    def test_sixteen_bit_crop_preserves_model_grid(self):
        source = np.full((94, 116), 52000, dtype=np.uint16)
        source[32:73, 50:67] = 2600
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "template.tif"
            out = Path(tmp) / "binary"
            Image.fromarray(source).save(image)
            args = SimpleNamespace(image=image, roi=(26, 20, 64, 64), polarity="dark",
                                   sigma=0, threshold=0.5, out=out, nm_per_px=2)
            run_segmentation(args)
            mask = read_mask(out / "template.png")
            np.testing.assert_array_equal(mask, source[20:84, 26:90] == 2600)
            self.assertTrue((out / "overlay.png").is_file())
            self.assertTrue((out / "segmentation.json").is_file())


if __name__ == "__main__":
    unittest.main()
