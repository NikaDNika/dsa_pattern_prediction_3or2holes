"""Compact U-Net with an auxiliary multiclass defect head."""

import torch
from torch import nn
from torch.nn import functional as F


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(4, cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(4, cout), nn.SiLU(),
    )


class PatternUNet(nn.Module):
    def __init__(self, width, classes):
        super().__init__()
        if width < 4 or width % 4:
            raise ValueError("width must be a multiple of 4")
        self.encoder = nn.ModuleList([
            conv_block(2, width), conv_block(width, 2 * width),
            conv_block(2 * width, 4 * width),
        ])
        self.bottom = conv_block(4 * width, 8 * width)
        self.decoder = nn.ModuleList([
            conv_block(12 * width, 4 * width), conv_block(6 * width, 2 * width),
            conv_block(3 * width, width),
        ])
        self.mask_head = nn.Conv2d(width, 1, 1)
        self.defect_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(32 * width, 4 * width),
            nn.SiLU(), nn.Dropout(0.15), nn.Linear(4 * width, classes),
        )

    def forward(self, image):
        skips = []
        for encode in self.encoder:
            image = encode(image)
            skips.append(image)
            image = F.max_pool2d(image, 2)
        image = self.bottom(image)
        class_logits = self.defect_head(image)
        for decode, skip in zip(self.decoder, reversed(skips)):
            image = F.interpolate(image, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            image = decode(torch.cat((image, skip), dim=1))
        return self.mask_head(image), class_logits


def multitask_loss(output, target_mask, target_class, class_weights, class_weight):
    mask_logits, classes = output
    probabilities = mask_logits.sigmoid().flatten(1)
    truth = target_mask.flatten(1)
    overlap = 2 * (probabilities * truth).sum(dim=1)
    dice = (overlap + 1) / (probabilities.sum(dim=1) + truth.sum(dim=1) + 1)
    seg = F.binary_cross_entropy_with_logits(mask_logits, target_mask) + (1 - dice.mean())
    clf = F.cross_entropy(classes, target_class, weight=class_weights)
    return seg + class_weight * clf
