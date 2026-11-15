"""ResNet-18 and controlled depthwise/SE variants for 43-class GTSRB.

All variants use the original 7x7/stride-2 stem and max-pooling, followed by
BasicBlock stages with [2, 2, 2, 2] blocks and [64, 128, 256, 512] channels.
Only the class count changes in the baseline. No pretrained weights are used.

Depthwise replacement: each residual-branch 3x3 convolution becomes a 3x3
depthwise convolution followed immediately by a 1x1 pointwise convolution.
There is NO extra BN or activation between these two operations. This keeps
the original outer BN/ReLU placement fixed across the convolution ablation.

SE placement: after the second residual-branch BN, before shortcut addition.
The stem and projection shortcuts remain standard convolutions in all groups.

Architecture references:
https://docs.pytorch.org/vision/0.23/_modules/torchvision/models/resnet.html
https://arxiv.org/abs/1512.03385
https://arxiv.org/abs/1704.04861
https://arxiv.org/abs/1709.01507
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


VARIANTS = ("baseline", "se", "depthwise", "depthwise_se")


class DepthwiseSeparableConv(nn.Module):
    """A linear depthwise 3x3 + pointwise 1x1 replacement for one 3x3 conv."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 3, stride=stride, padding=1,
            groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class SqueezeExcitation(nn.Module):
    """Global average pooling -> C/16 -> C -> sigmoid channel gating."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Bias-free FC layers are an explicit fixed implementation choice.
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = self.avgpool(x).flatten(1)
        scale = self.sigmoid(self.fc2(self.relu(self.fc1(scale))))
        return x * scale[:, :, None, None]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1,
        downsample: nn.Module | None = None, *, depthwise: bool = False,
        use_se: bool = False,
    ):
        super().__init__()
        if depthwise:
            self.conv1 = DepthwiseSeparableConv(inplanes, planes, stride)
            self.conv2 = DepthwiseSeparableConv(planes, planes)
        else:
            self.conv1 = nn.Conv2d(
                inplanes, planes, 3, stride=stride, padding=1, bias=False,
            )
            self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SqueezeExcitation(planes) if use_se else nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class TrafficResNet18(nn.Module):
    """ResNet-18 with independently configurable convolution and SE factors."""

    def __init__(self, num_classes: int = 43, *, depthwise: bool = False, use_se: bool = False):
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        self.inplanes = 64
        self.depthwise = depthwise
        self.use_se = use_se
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(128, stride=2)
        self.layer3 = self._make_layer(256, stride=2)
        self.layer4 = self._make_layer(512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, planes: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        first = BasicBlock(
            self.inplanes, planes, stride, downsample,
            depthwise=self.depthwise, use_se=self.use_se,
        )
        self.inplanes = planes
        second = BasicBlock(
            planes, planes, depthwise=self.depthwise, use_se=self.use_se,
        )
        return nn.Sequential(first, second)

    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


def build_model(variant: str, num_classes: int = 43) -> TrafficResNet18:
    """Build an untrained model; control the random seed before calling."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}")
    return TrafficResNet18(
        num_classes=num_classes,
        depthwise=variant in ("depthwise", "depthwise_se"),
        use_se=variant in ("se", "depthwise_se"),
    )


def profile_model(
    model: nn.Module, input_size: tuple[int, int, int, int] = (1, 3, 64, 64),
) -> dict[str, Any]:
    """Count parameters and per-image Conv2d/Linear MACs using forward hooks.

    A multiply-accumulate is one MAC (roughly two FLOPs). This intentionally
    excludes bias addition, BN, pooling, ReLU/sigmoid, residual addition and SE
    elementwise multiplication. The SE FC layers ARE counted. Parameter count
    includes learned BN and classifier bias parameters, but excludes buffers.
    All prior per-module train/eval states are restored, including mixed states.
    The profiling pass runs in eval/inference mode and never modifies BN stats.
    """
    if len(input_size) != 4 or input_size[0] < 1 or input_size[1] != 3:
        raise ValueError("input_size must be a positive NCHW RGB shape")
    first_param = next(model.parameters())
    prior_states = [(module, module.training) for module in model.modules()]
    handles = []
    total_macs = 0

    def count(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        nonlocal total_macs
        if isinstance(module, nn.Conv2d):
            per_output = (
                module.in_channels // module.groups
                * module.kernel_size[0] * module.kernel_size[1]
            )
            total_macs += output.numel() * per_output
        elif isinstance(module, nn.Linear):
            total_macs += output.numel() * module.in_features

    try:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handles.append(module.register_forward_hook(count))
        model.eval()
        with torch.inference_mode():
            sample = torch.zeros(input_size, device=first_param.device, dtype=first_param.dtype)
            output = model(sample)
        result = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "macs": total_macs // input_size[0],
            "input_size": list(input_size),
            "output_size": list(output.shape),
            "macs_scope": "Conv2d and Linear only; one MAC per multiply-accumulate",
        }
    finally:
        for handle in handles:
            handle.remove()
        for module, training in prior_states:
            module.training = training
    return result


if __name__ == "__main__":
    import json
    torch.set_num_threads(4)
    print(json.dumps({name: profile_model(build_model(name)) for name in VARIANTS}, indent=2))
