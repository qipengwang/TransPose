import sys
sys.path.append('lib')

from models.mobilenet import MobileNetV1, MobileNetV2, MobileNetV3_Small, MobileNetV3_Large

net = MobileNetV1()
print(len(net.features))
net = MobileNetV2()
print(len(net.features))
net = MobileNetV3_Large()
print(len(net.features))
net = MobileNetV3_Small()
print(len(net.features))