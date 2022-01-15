import torch
import torch.nn as nn

class LinearProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = [nn.Conv2d(3, 3, 8, 8)]
        self.model = nn.Sequential(*self.features)
        self.features_out = [3]
    
    def forward(self, x):
        return self.model(x)