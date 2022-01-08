import warnings
from collections import namedtuple
from typing import Optional, Tuple, List, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Inception(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
    ) -> None:
        super().__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1), 
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, ) -> None:
        super().__init__()
        self.features = [
            BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3),  # N x 64 x 112 x 112
            nn.MaxPool2d(3, stride=2, ceil_mode=True),  # N x 64 x 56 x 56
            BasicConv2d(64, 64, kernel_size=1),  # N x 64 x 56 x 56
            BasicConv2d(64, 192, kernel_size=3, padding=1),  # N x 192 x 56 x 56
            nn.MaxPool2d(3, stride=2, ceil_mode=True),  # N x 192 x 56 x 56

            Inception(192, 64, 96, 128, 16, 32, 32),  # N x 256 x 28 x 28
            Inception(256, 128, 128, 192, 32, 96, 64),  # N x 480 x 28 x 28
            nn.MaxPool2d(3, stride=2, ceil_mode=True),  # N x 480 x 14 x 14

            Inception(480, 192, 96, 208, 16, 48, 64),  # N x 512 x 14 x 14
            Inception(512, 160, 112, 224, 24, 64, 64),  # N x 512 x 14 x 14
            Inception(512, 128, 128, 256, 24, 64, 64),  # N x 512 x 14 x 14
            Inception(512, 112, 144, 288, 32, 64, 64),  # N x 528 x 14 x 14
            Inception(528, 256, 160, 320, 32, 128, 128),  # N x 832 x 14 x 14
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # N x 832 x 7 x 7

            Inception(832, 256, 160, 320, 32, 128, 128),  # N x 1024 x 7 x 7
            Inception(832, 384, 192, 384, 48, 128, 128),  # N x 1024 x 7 x 7
        ]
        self.features_out = [
            64, 64, 64, 192, 192,
            256, 480, 480,
            512, 512, 512, 528, 832, 832,
            1024, 1024
        ]
        self.model = nn.Sequential(*self.features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)),
        self.dropout = nn.Dropout(dropout),
        self.fc = nn.Linear(1024, num_classes),
        

    def forward(self, x: Tensor):
        x = self.model(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


