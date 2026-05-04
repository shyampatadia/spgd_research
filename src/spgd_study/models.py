"""Network factories for Experiments 2-4.

Two Moons (Exp 2) and OpenML tabular (Exp 3) use the configurable MLP below.
A ResNet-18 wrapper for CIFAR-10 (Exp 4) is added in Step 5.
"""

from __future__ import annotations

import torch.nn as nn


class MLP(nn.Module):
    """Configurable feed-forward network.

    Layout: in_dim -> hidden -> hidden -> ... -> out_dim
    n_hidden = number of hidden Linear layers (so n_hidden=2 means
    in -> hidden -> hidden -> out). ReLU activations, no BatchNorm
    (proposal explicit -- BN can mask saddle structure).

    For binary classification (out_dim == 1) the forward pass squeezes the
    trailing singleton so output shape matches typical 1D label tensors.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_hidden: int = 2):
        super().__init__()
        if n_hidden < 1:
            raise ValueError("n_hidden must be >= 1")
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x):
        out = self.net(x)
        if self.out_dim == 1:
            out = out.squeeze(-1)
        return out


def two_moons_mlp(hidden: int = 32) -> MLP:
    """Default Two Moons architecture per proposal: 2 -> hidden -> hidden -> 1."""
    return MLP(in_dim=2, hidden=hidden, out_dim=1, n_hidden=2)


def cifar10_resnet18(
    disable_bn_layers=("layer1", "layer2"),
    num_classes: int = 10,
):
    """CIFAR-style ResNet-18 with BatchNorm disabled in selected stages.

    Differences from torchvision's ImageNet ResNet-18:
        - First conv replaced with 3x3 stride-1 (input is 32x32, not 224x224).
        - First maxpool removed.
        - BN replaced with Identity in the residual stages listed in
          ``disable_bn_layers``. Default (layer1, layer2) preserves saddle
          structure in the early features per the proposal, while keeping
          BN in deeper stages so the network is still trainable.

    Set ``disable_bn_layers=()`` to keep BN everywhere (the standard CIFAR
    baseline) or ``disable_bn_layers=("layer1","layer2","layer3","layer4")``
    to remove BN entirely (max saddle structure, hardest to train).
    """
    from torch import nn
    from torchvision.models import resnet18

    net = resnet18(num_classes=num_classes)
    # CIFAR-friendly stem
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()

    def replace_bn(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                setattr(module, name, nn.Identity())
            else:
                replace_bn(child)

    for layer_name in disable_bn_layers:
        if hasattr(net, layer_name):
            replace_bn(getattr(net, layer_name))

    return net
