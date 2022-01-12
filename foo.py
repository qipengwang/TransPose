import sys
import torch.nn as nn
import torch
sys.path.append('lib')

from models.mobilenet import MobileNetV1, MobileNetV2, MobileNetV3_Small, MobileNetV3_Large
from models.googlenet import GoogLeNet
from models.xception import Xception
from models.inceptionv3 import InceptionV3
from models.shufflenetv2 import ShuffleNetV2
from models.squeezenet import SqueezeNet
from models.vgg import *
from models.linear import LinearProjection
from models.swin_transpose import SwinTransformer

# import os
# root = 'experiments/coco/swin_transpose'
# for file in os.listdir(root):
#     with open(os.path.join(root, file)) as f:
#         lines = []
#         for line in f:
#             lines.append(line.replace('transpose_cv', 'swin_transpose'))
#     with open(os.path.join(root, 'S'+file), 'w') as f:
#         f.writelines(lines)

ic = 3
st = SwinTransformer(input_shape=(32, 24), input_channel=ic)
x = torch.randn(2, ic, 32, 24)
print(st(x).size())
exit(0)

# B, H, W, C = 2, 6, 4, 3
# x = torch.arange(B * H * W * C).reshape([B, H, W, C])
# print(x, x.shape)
# x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
# x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
# x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
# x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
# x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
# x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
# # print(x, x.shape)
# B, L, C = x.shape
# H, W = H//2, W//2
# x = x.reshape([B, H, W, 2, 2, C//4]).permute([0, 1, 4, 2, 3, 5]).reshape([B, H*2, W*2, C//4])
# print(x, x.shape)