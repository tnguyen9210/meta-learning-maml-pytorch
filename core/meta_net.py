
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import core.meta_modules as meta


class MetaConvNet(meta.MetaModule):
    def __init__(self, args):
        super(MetaConvNet, self).__init__()

        # conv layers ##
        widen_factor = args['conv_widen_factor']
        droprate = args['conv_droprate']
        nchannels = [16, 16, 32, 32, 32]
        nchannels = [num*widen_factor for num in nchannels]  # widen the network
        
        self.block1 = self.make_block(nchannels[0], nchannels[1])
        self.block2 = self.make_block(nchannels[1], nchannels[2])
        self.block3 = self.make_block(nchannels[2], nchannels[3])
        self.block3 = self.make_block(nchannels[3], nchannels[4])
        
        # fc layers ##
        in_dim = nchannels[-1]
        num_classes = args['nway']
        self.img_cls = meta.MetaLinear(in_dim, num_classes)
        
    def forward(self, x):
        
        # conv layers
        hidden = self.block1(hidden)
        hidden = self.block2(hidden)
        hidden = self.block3(hidden)
        hidden = self.block4(hidden)
        hidden = hidden.view(hidden.shape[0], -1)
        
        # fc layers
        logit = self.img_cls(hidden)

        return logit

    def make_block(self, in_channels, out_channels):
        return meta.MetaSequential(
            meta.MetaConv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            meta.MetaBatchNorm2d(
                out_channels, momentum=1., track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    


