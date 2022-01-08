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

# net = MobileNetV1()
# print(net.features_out)
# net = GoogLeNet()
# print(net.features_out)
# net = Xception()
# print(net.features_out)
net = ShuffleNetV2()
net = nn.Sequential(
    *net.features[:5]
)
x = torch.randn(2, 3, 256, 192)
print(net(x).size())
# net = ShuffleNetV2()
# print(net.features_out)
# net = SqueezeNet()
# print(net.features_out)