"""make_conv_layers / make_linear_layers / make_deconv_layers helpers.

Verbatim port of `third_party/SMPLer-X/common/nets/layer.py` — already
plain torch, no decorators or framework deps. Kept here to avoid a
sys.path injection dance into `third_party/SMPLer-X/common/nets/`.
"""

from __future__ import annotations

import torch.nn as nn


def make_linear_layers(feat_dims: list[int], relu_final: bool = True,
                       use_bn: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(feat_dims) - 1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i + 1]))
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def make_conv_layers(feat_dims: list[int], kernel: int = 3, stride: int = 1,
                     padding: int = 1, bnrelu_final: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i + 1],
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
        )
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def make_deconv_layers(feat_dims: list[int], bnrelu_final: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i + 1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False,
            )
        )
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)
