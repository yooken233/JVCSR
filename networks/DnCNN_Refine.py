import torch
import torch.nn as nn
import torch.nn.init as init
from .blocks import ConvBlock, DeconvBlock, MeanShift


        
class DnCNN_Refine(nn.Module):
    def __init__(self):
        super(DnCNN_Refine, self).__init__()
        # if sr == 0.9375:
            # d = 0
        # if sr == 0.875:
            # d = 1
        # elif sr == 0.75:
            # d = 2
        # elif sr == 0.5:
            # d = 3
        # elif sr == 0.25:
            # d = 4
        # elif sr == 0.125:
            # d = 5
        d=2
        depth=17
        n_channels=64
        image_channels=1
        kernel_size = 3
        padding = 1
        layers = []
        
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        
        
        for _ in range(depth-d):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn_r = nn.Sequential(*layers)
        
        
        
        self._initialize_weights()

    def forward(self, x):
        y = x
        
        out = self.dncnn_r(x)
        
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def reset_state(self):
        self.should_reset = True

