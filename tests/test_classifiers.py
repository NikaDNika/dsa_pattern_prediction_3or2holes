"""Check classifier interfaces and independent replacement of the classification head."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dsa.models.classifiers import build_classifier
from dsa.models.unet import PatternUNet
from dsa.training.classification import metrics
from dsa.training.reconstruction import predict_sem


class ClassifierTests(unittest.TestCase):
    def test_two_channel_model_interfaces(self):
        torch.set_num_threads(2)
        with torch.inference_mode():
            for name in ("unet_encoder", "resnet18", "convnext_tiny"):
                with self.subTest(model=name):
                    model=build_classifier(name).eval()
                    logits=model(torch.zeros(1,2,256,256))
                    self.assertEqual(tuple(logits.shape),(1,2))
                    self.assertTrue(torch.isfinite(logits).all())
                    del model

    def test_majority_accuracy_does_not_hide_minority_failure(self):
        scores=metrics(np.array([0,1,1,1]),np.tile([0.,1.],(4,1)))
        self.assertEqual(scores["accuracy"],.75)
        self.assertEqual(scores["two_recall"],0)
        self.assertEqual(scores["balanced_accuracy"],.5)

    def test_classifier_override_preserves_grayscale_output(self):
        torch.set_num_threads(2)
        generator=PatternUNet(8,2).eval()
        classifier=build_classifier("unet_encoder").eval()
        with torch.no_grad():
            classifier.head[-1].weight.zero_()
            classifier.head[-1].bias.copy_(torch.tensor([10.,-10.]))
        state={"epoch":1}
        combined=state|{"template_classifier":{"model":"unet_encoder","weights":classifier.state_dict()}}
        x=np.zeros((2,256,256),dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            predict_sem(generator,state,torch.device("cpu"),x,root/"old",.7)
            result=predict_sem(generator,combined,torch.device("cpu"),x,root/"new",.7)
            self.assertEqual(result["prediction"],"2孔")
            self.assertEqual(result["classifier_model"],"unet_encoder")
            with Image.open(root/"old/predicted_sem.png") as a, Image.open(root/"new/predicted_sem.png") as b:
                np.testing.assert_array_equal(np.asarray(a),np.asarray(b))


if __name__=="__main__":unittest.main()
