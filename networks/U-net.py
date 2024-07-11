import torch
import torch.nn as nn
import torch.nn.init as init
from .blocks import ConvBlock, DeconvBlock, MeanShift
from networks.TransitionBlock import TransitionBlock
from networks.DownBlock import DownBlock
from networks.DenseBlock import DenseBlock
from networks.UpBlock import UpBlock


class FeedbackBlock(nn.Module):
    def __init__(self, num_features, num_groups, upscale_factor, act_type, norm_type):
        super(FeedbackBlock, self).__init__()
        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            kernel_size = 12

        self.num_groups = num_groups

        self.compress_in = ConvBlock(2*num_features, num_features,
                                     kernel_size=1,
                                     act_type=act_type, norm_type=norm_type)

        self.upBlocks = nn.ModuleList()
        self.downBlocks = nn.ModuleList()
        self.uptranBlocks = nn.ModuleList()
        self.downtranBlocks = nn.ModuleList()

        for idx in range(self.num_groups):
            self.upBlocks.append(DeconvBlock(num_features, num_features,
                                             kernel_size=kernel_size, stride=stride, padding=padding,
                                             act_type=act_type, norm_type=norm_type))
            self.downBlocks.append(ConvBlock(num_features, num_features,
                                             kernel_size=kernel_size, stride=stride, padding=padding,
                                             act_type=act_type, norm_type=norm_type, valid_padding=False))
            if idx > 0:
                self.uptranBlocks.append(ConvBlock(num_features*(idx+1), num_features,
                                                   kernel_size=1, stride=1,
                                                   act_type=act_type, norm_type=norm_type))
                self.downtranBlocks.append(ConvBlock(num_features*(idx+1), num_features,
                                                     kernel_size=1, stride=1,
                                                     act_type=act_type, norm_type=norm_type))

        self.compress_out = ConvBlock(num_groups*num_features, num_features,
                                      kernel_size=1,
                                      act_type=act_type, norm_type=norm_type)

        self.should_reset = True
        self.last_hidden = None

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)
        x = self.compress_in(x)

        lr_features = []
        hr_features = []
        lr_features.append(x)

        for idx in range(self.num_groups):
            LD_L = torch.cat(tuple(lr_features), 1)    # when idx == 0, lr_features == [x]
            if idx > 0:
                LD_L = self.uptranBlocks[idx-1](LD_L)
            LD_H = self.upBlocks[idx](LD_L)

            hr_features.append(LD_H)

            LD_H = torch.cat(tuple(hr_features), 1)
            if idx > 0:
                LD_H = self.downtranBlocks[idx-1](LD_H)
            LD_L = self.downBlocks[idx](LD_H)

            lr_features.append(LD_L)

        del hr_features
        output = torch.cat(tuple(lr_features[1:]), 1)   # leave out input x, i.e. lr_features[0]
        output = self.compress_out(output)

        self.last_hidden = output

        return output

    def reset_state(self):
        self.should_reset = True
        
class conv_block(nn.Module):

    """

    Convolution Block 

    """

    def __init__(self, in_ch, out_ch):

        super(conv_block, self).__init__()

        

        self.conv = nn.Sequential(

            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True))



    def forward(self, x):



        x = self.conv(x)

        return x





class up_conv(nn.Module):

    """

    Up Convolution Block

    """

    def __init__(self, in_ch, out_ch):

        super(up_conv, self).__init__()

        self.up = nn.Sequential(

            nn.Upsample(scale_factor=2),

            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True)

        )



    def forward(self, x):

        x = self.up(x)

        return x





class U_Net(nn.Module):

    """

    UNet - Basic Implementation

    Paper : https://arxiv.org/abs/1505.04597

    """

    def __init__(self, in_ch=3, out_ch=3):

        super(U_Net, self).__init__()



        n1 = 64

        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2,padding=)



        self.Conv1 = conv_block(in_ch, filters[0])

        self.Conv2 = conv_block(filters[0], filters[1])

        self.Conv3 = conv_block(filters[1], filters[2])

        self.Conv4 = conv_block(filters[2], filters[3])

        self.Conv5 = conv_block(filters[3], filters[4])



        self.Up5 = up_conv(filters[4], filters[3])

        self.Up_conv5 = conv_block(filters[4], filters[3])



        self.Up4 = up_conv(filters[3], filters[2])

        self.Up_conv4 = conv_block(filters[3], filters[2])



        self.Up3 = up_conv(filters[2], filters[1])

        self.Up_conv3 = conv_block(filters[2], filters[1])



        self.Up2 = up_conv(filters[1], filters[0])

        self.Up_conv2 = conv_block(filters[1], filters[0])



        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)



       # self.active = torch.nn.Sigmoid()



    def forward(self, x):



        e1 = self.Conv1(x)



        e2 = self.Maxpool1(e1)

        e2 = self.Conv2(e2)



        e3 = self.Maxpool2(e2)

        e3 = self.Conv3(e3)



        e4 = self.Maxpool3(e3)

        e4 = self.Conv4(e4)



        e5 = self.Maxpool4(e4)

        e5 = self.Conv5(e5)



        d5 = self.Up5(e5)

        d5 = torch.cat((e4, d5), dim=1)



        d5 = self.Up_conv5(d5)



        d4 = self.Up4(d5)

        d4 = torch.cat((e3, d4), dim=1)

        d4 = self.Up_conv4(d4)



        d3 = self.Up3(d4)

        d3 = torch.cat((e2, d3), dim=1)

        d3 = self.Up_conv3(d3)



        d2 = self.Up2(d3)

        d2 = torch.cat((e1, d2), dim=1)

        d2 = self.Up_conv2(d2)



        out = self.Conv(d2)



        #d1 = self.active(out)



        return out   
        

        
class SRFBN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_steps, num_groups, upscale_factor, act_type = 'prelu', norm_type = None):
        super(SRFBN, self).__init__()

        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            kernel_size = 12

        self.num_steps = num_steps
        self.num_features = num_features
        self.upscale_factor = upscale_factor

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)
        
        self.refine1 = U_Net()
        
        # LR feature extraction block
        self.conv_in = ConvBlock(in_channels, 4*num_features,
                                 kernel_size=3,
                                 act_type=act_type, norm_type=norm_type)
        
      
                                 
        self.feat_in = ConvBlock(4*num_features, num_features,
                                 kernel_size=1,
                                 act_type=act_type, norm_type=norm_type)

        # basic block
        self.block = FeedbackBlock(num_features, num_groups, upscale_factor, act_type, norm_type)

        # reconstruction block
		# uncomment for pytorch 0.4.0
        # self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')

        self.out = DeconvBlock(num_features, num_features,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               act_type='prelu', norm_type=norm_type)
       
        self.conv_out = ConvBlock(num_features, out_channels,
                                  kernel_size=3,
                                  act_type=None, norm_type=norm_type)
            
        #self.refine2 = DnCNN_Refine()
        
        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

    def forward(self, x):
        self._reset_state()

        x = self.sub_mean(x)
		# uncomment for pytorch 0.4.0
        # inter_res = self.upsample(x)
		
     
		# comment for pytorch 0.4.0
        inter_res = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        
        # RefineNet
        x = self.refine1(x)
        
        x = self.conv_in(x)
        x = self.feat_in(x)

        outs = []
        for _ in range(self.num_steps):
            h = self.block(x)
           
            h = torch.add(inter_res, self.conv_out(self.out(h)))
            # RefineNet
            #h = self.refine2(h)
            h = self.add_mean(h)
            outs.append(h)

        return outs # return output of every timesteps

    def _reset_state(self):
        self.block.reset_state()