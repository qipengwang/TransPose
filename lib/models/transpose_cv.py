# ------------------------------------------------------------------------------
# Copyright (c) Southeast University. Licensed under the MIT License.
# Written by Sen Yang (yangsenius@seu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import math
from posixpath import basename

import torch
from torch.nn import init
import torch.nn.functional as F
from torch import nn, Tensor
from collections import OrderedDict

import copy
from typing import Optional, List

from .transpose_r import TransformerEncoder, TransformerEncoderLayer
from .mobilenet import MobileNetV1, MobileNetV2, MobileNetV3_Large, MobileNetV3_Small
from .googlenet import GoogLeNet
from .xception import Xception
from .inceptionv3 import InceptionV3
from .shufflenetv2 import ShuffleNetV2
from .squeezenet import SqueezeNet
from .vgg import *
from .linear import LinearProjection


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class TransPoseM(nn.Module):

    def __init__(self, basenet, num_features, cfg, **kwargs):
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(TransPoseM, self).__init__()
        # base net
        model = basenet()
        self.features = model.features[:num_features]
        # print(self.features[-1])
        self.features = nn.Sequential(*self.features)
        num_out = model.features_out[num_features - 1]
        # print(num_out)

        d_model = cfg.MODEL.DIM_MODEL
        dim_feedforward = cfg.MODEL.DIM_FEEDFORWARD
        encoder_layers_num = cfg.MODEL.ENCODER_LAYERS
        n_head = cfg.MODEL.N_HEAD
        pos_embedding_type = cfg.MODEL.POS_EMBEDDING
        w, h = cfg.MODEL.IMAGE_SIZE

        self.reduce = nn.Conv2d(num_out, d_model, 1, bias=False)
        self._make_position_embedding(w, h, d_model, pos_embedding_type)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            activation='relu',
            return_atten_map=False
        )
        self.global_encoder = TransformerEncoder(
            encoder_layer,
            encoder_layers_num,
            return_atten_map=False
        )

        # used for deconv layers
        self.inplanes = d_model
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,   # 1
            extra.NUM_DECONV_FILTERS,  # [d_model]
            extra.NUM_DECONV_KERNELS,  # [4]
        )

        self.final_layer = nn.Conv2d(
            in_channels=d_model,
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding = None
            logger.info("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h // 8
                self.pe_w = w // 8
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(
                    torch.randn(length, 1, d_model))
                logger.info("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                logger.info("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2*math.pi):
        # logger.info(">> NOTE: this is for testing on unseen input resolutions")
        # # NOTE generalization test with interploation
        # self.pe_h, self.pe_w = 256 // 8 , 192 // 8 #self.pe_h, self.pe_w
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(2, 0, 1)
        return pos  # [h*w, 1, d_model]

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.reduce(x)
        # print(x.size())

        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x = self.global_encoder(x, pos=self.pos_embedding)
        x = x.permute(1, 2, 0).contiguous().view(bs, c, h, w)
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init final conv weights from normal distribution')
            for name, m in self.final_layer.named_modules():
                if isinstance(m, nn.Conv2d):
                    logger.info(
                        '=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            existing_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name in self.state_dict():
                    existing_state_dict[name] = m
                    print(":: {} is loaded from {}".format(name, pretrained))
            self.load_state_dict(existing_state_dict, strict=False)
        else:
            logger.info('=> NOTE :: ImageNet Pretrained Weights {} are not loaded ! Please Download it'.format(pretrained))
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)



def get_pose_net(cfg, is_train, **kwargs):
    # stride = [2, 2, 2]
    mobilenet_spec = {
        'MobileNet': (MobileNetV1, 6),  # 14
        'MobileNetV1': (MobileNetV1, 6),  # 14
        'MobileNetV2': (MobileNetV2, 7),  # 19
        'MobileNetV3Large': (MobileNetV3_Large, 9),  # 21
        'MobileNetV3_Large': (MobileNetV3_Large, 9),  # 21
        'MobileNetV3Small': (MobileNetV3_Small, 6),  # 17
        'MobileNetV3_Small': (MobileNetV3_Small, 6),  # 17
        'GoogleNet': (GoogLeNet, 7),
        'Xception': (Xception, 8),
        'InceptionV3': (InceptionV3, 10),
        'ShuffleNetV2': (ShuffleNetV2, 5),
        'SqueezeNet': (SqueezeNet, 12),
        'SqueezeNetV1.0': (SqueezeNet, 12),
        'VGG11': (VGG11, 9),
        'VGG13': (VGG13, 11),
        'VGG16': (VGG16, 13),
        'VGG19': (VGG19, 15),
        'Linear': (LinearProjection, 1),
        'LinearProjection': (LinearProjection, 1),

    }
    basenet, num_features = mobilenet_spec[kwargs['backbone']]
    model = TransPoseM(basenet, num_features, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
