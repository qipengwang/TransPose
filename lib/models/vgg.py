from typing import Union, List, Dict, Any

import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features: nn.Module, features_out: list, num_classes: int = 1000, dropout: float = 0.5):
        super().__init__()
        self.features = features
        self.features_out = features_out
        self.model = nn.Sequential(*self.features)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]]):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [nn.Sequential(conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True))]
            in_channels = v
    return layers


def _vgg(cfg: str, **kwargs: Any):
    cfgs: Dict[str, List[Union[str, int]]] = {
        11: [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        13: [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        16: [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
        19: [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
    }
    feature_out = cfgs[cfg].copy()
    for i in range(len(feature_out)):
        if feature_out[i] == 'M':
            feature_out[i] = feature_out[i-1]
    return VGG(make_layers(cfgs[cfg]), feature_out, **kwargs)

def VGG11(): return _vgg(11)
def VGG13(): return _vgg(13)
def VGG16(): return _vgg(16)
def VGG19(): return _vgg(19)
