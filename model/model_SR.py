# -*- encoding: utf-8 -*-
from torch import optim
from psrt import Block
import torch
import torch.nn as nn


def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):   ## initialization for Conv2d
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):   ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):     ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

def init_w(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # torch.nn.init.uniform_(m.weight, a=0, b=1)
            elif isinstance(m, nn.Conv2d):   ## initialization for Conv2d
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

class PSRTnet(nn.Module):
    def __init__(self, args):
        super(PSRTnet, self).__init__()
        self.args = args
        self.img_size = 64
        self.in_channels = 31
        self.embed = 32
        self.conv = nn.Sequential(
            nn.Conv2d(self.embed, self.in_channels, 3, 1, 1), nn.LeakyReLU(0.2, True)
        )
        # self.w = Block(num=2, img_size=self.img_size, in_chans=34, embed_dim=32, head=8, win_size=2)
        self.w = Block(out_num=2, inside_num=3, img_size=self.img_size, in_chans=34, embed_dim=self.embed, head=8,
                       win_size=8)
        self.visual_corresponding_name = {}
        init_weights(self.conv)
        init_w(self.w)

    def forward(self, rgb, lms):
        '''
        :param rgb:
        :param lms:
        :return:
        '''
        self.rgb = rgb
        self.lms = lms
        xt = torch.cat((self.lms, self.rgb), 1)  # Bx34X64x64
        _, _, H, W = xt.shape
        w_out = self.w(H, W, xt)
        self.result = self.conv(w_out) + self.lms

        return self.result

    def name(self):
        return 'PSRT'

