"""Two-channel U-Net encoder, ResNet-18 and ConvNeXt-Tiny classifiers."""
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, convnext_tiny
from .unet import PatternUNet


def build_classifier(name):
    if name == "unet_encoder":
        class EncoderClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                u=PatternUNet(8,2)
                self.encoder=u.encoder
                self.bottom=u.bottom
                self.head=u.defect_head
            def forward(self,x):
                for block in self.encoder:
                    x=F.max_pool2d(block(x),2)
                return self.head(self.bottom(x))
        return EncoderClassifier()
    if name == "resnet18":
        model=resnet18(weights=None)
        model.conv1=nn.Conv2d(2,64,7,2,3,bias=False)
        model.fc=nn.Linear(512,2)
        return model
    if name == "convnext_tiny":
        model=convnext_tiny(weights=None,stochastic_depth_prob=.1)
        model.features[0][0]=nn.Conv2d(2,96,4,4)
        model.classifier[2]=nn.Linear(768,2)
        return model
    raise ValueError(name)
