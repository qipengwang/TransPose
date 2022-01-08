import warnings
from collections import namedtuple
from typing import Callable, Any, Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.modules import padding

from .googlenet import BasicConv2d


class InceptionA(nn.Module):
    def __init__(self, in_channels: int, pool_features: int):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x: Tensor):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat([branch1x1, branch5x5, branch3x3dbl, branch_pool], 1)
        


class InceptionB(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x: Tensor):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        return torch.cat([branch3x3, branch3x3dbl, branch_pool], 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels: int, channels_7x7: int):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        self.branch7x7_1 = BasicConv2d(in_channels, channels_7x7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(channels_7x7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, channels_7x7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(channels_7x7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x: Tensor):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat([branch1x1, branch7x7, branch7x7dbl, branch_pool], 1)



class InceptionD(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x: Tensor):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        return torch.cat([branch3x3, branch7x7x3, branch_pool], 1)


class InceptionE(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x: Tensor):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        return torch.cat([branch1x1, branch3x3, branch3x3dbl, branch_pool], 1)
        

class InceptionV3(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super().__init__()
        self.features = [
            BasicConv2d(3, 32, kernel_size=3, stride=2, padding=1),
            BasicConv2d(32, 32, kernel_size=3, padding='same'),
            BasicConv2d(32, 64, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), 

            BasicConv2d(64, 80, kernel_size=1),
            BasicConv2d(80, 192, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            InceptionA(192, pool_features=32),
            InceptionA(256, pool_features=64),
            InceptionA(288, pool_features=64),

            InceptionB(288),  # stride=2
            InceptionC(768, channels_7x7=128),
            InceptionC(768, channels_7x7=160),
            InceptionC(768, channels_7x7=160),
            InceptionC(768, channels_7x7=192),

            InceptionD(768),  # stride=2
            InceptionE(1280),
            InceptionE(2048),
        ]
        self.features_out = [
            32, 32, 64, 64,
            80, 192, 192,
            256, 288, 288,
            768, 768, 768, 768, 768,
            1280, 2048, 2048
        ]
        self.model = nn.Sequential(*self.features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: Tensor):
        x = self.model(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
