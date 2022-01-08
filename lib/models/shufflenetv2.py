from typing import Callable, Any, List

import torch
import torch.nn as nn
from torch import Tensor


def channel_shuffle(x: Tensor, groups: int):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int) -> None:
        super().__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=3, stride=self.stride, padding=1, groups=inp),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features, kernel_size=1, stride=1, padding=0, bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1, groups=branch_features),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        stages_repeats: List[int] = [4, 8, 4],
        stages_out_channels: List[int] = [24, 116, 232, 464, 1024],
        num_classes: int = 1000,
    ):
        super().__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]

        self.features, self.features_out = [], []

        self.features = [
            nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        ]
        self.features_out.append(output_channels)

        input_channels = output_channels
        for repeats, output_channels in zip(stages_repeats, self._stage_out_channels[1:]):
            self.features.append(InvertedResidual(input_channels, output_channels, 2))
            self.features_out.append(output_channels)
            for _ in range(repeats - 1):
                self.features.append(InvertedResidual(output_channels, output_channels, 1))
                self.features_out.append(output_channels)
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.features.append(
            nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        )
        self.features_out.append(output_channels)
        self.model = nn.Sequential(*self.features)

        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x
