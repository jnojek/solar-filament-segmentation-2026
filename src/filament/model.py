"""Compact two-convolution-per-level U-Net for single-channel H-alpha images."""

import torch
from torch import nn
from torch.nn import functional as F


def double_conv(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
        nn.GroupNorm(8, outputs),
        nn.SiLU(inplace=True),
        nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
        nn.GroupNorm(8, outputs),
        nn.SiLU(inplace=True),
    )


class FilamentUNet(nn.Module):
    def __init__(self, channels=24):
        super().__init__()
        widths = [channels * 2**level for level in range(5)]
        self.encoder = nn.ModuleList(
            double_conv(previous, width) for previous, width in zip([1] + widths, widths)
        )
        self.up = nn.ModuleList(
            nn.ConvTranspose2d(deep, shallow, 2, stride=2)
            for deep, shallow in zip(reversed(widths[1:]), reversed(widths[:-1]))
        )
        self.decoder = nn.ModuleList(double_conv(width * 2, width) for width in reversed(widths[:-1]))
        self.output = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x):
        skips = []
        for block in self.encoder[:-1]:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.encoder[-1](x)
        for up, decode, skip in zip(self.up, self.decoder, reversed(skips)):
            x = decode(torch.cat((up(x), skip), dim=1))
        return self.output(x)


def filament_loss(logits, target):
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    # Upweight sparse positive pixels. Tile-level Dice supplies a second shape term.
    bce = (bce * (1 + 5 * target)).mean()
    probability = logits.sigmoid()
    axes = (1, 2, 3)
    overlap = (probability * target).sum(axes)
    dice = (2 * overlap + 1) / (probability.sum(axes) + target.sum(axes) + 1)
    return bce + 1 - dice.mean()
