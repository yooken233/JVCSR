import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.autograd import Variable
from .blocks import ConvBlock, DeconvBlock, MeanShift
     
class EDSR(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_blocks, res_scale, upscale_factor):
        super(EDSR, self).__init__()
        
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)
        
        self.conv1 = ConvBlock(3, 64,
                                 kernel_size=9,padding=4,
                                 act_type='relu', norm_type=None)
        self.conv2 = ConvBlock(64, 32,
                                 kernel_size=1,padding=0,
                                 act_type='relu', norm_type=None)   
        self.conv3 = ConvBlock(32, 3,
                                 kernel_size=5,padding=2,
                                 act_type=None, norm_type=None)
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5,padding=2)
        # self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out