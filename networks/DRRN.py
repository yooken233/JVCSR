import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.autograd import Variable

class DRRN(nn.Module):
	def __init__(self):
		super(DRRN, self).__init__()
		self.input = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
		self.output = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
		self.relu = nn.ReLU(inplace=True)

		# weights initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, sqrt(2. / n))

	def forward(self, x):
		residual = x
		inputs = self.input(self.relu(x))
		out = inputs
		for _ in range(25):
			out = self.conv2(self.relu(self.conv1(self.relu(out))))
			out = torch.add(out, inputs)

		out = self.output(self.relu(out))
		out = torch.add(out, residual)
		return out
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.autograd import Variable

class EDSR(nn.Module):
	def __init__(self, in_channels, out_channels, num_features, num_blocks, res_scale, upscale_factor, conv=default_conv):
		super(EDSR, self).__init__()
		self.input = nn.Conv2d(in_channels=in_channels, out_channels=num_features, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv1 = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv2 = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1, padding=1, bias=False)
		self.output = nn.Conv2d(in_channels=num_features, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.relu = nn.ReLU(inplace=True)

		# weights initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, sqrt(2. / n))

	def forward(self, x):
		residual = x
		inputs = self.input(self.relu(x))
		out = inputs
		for _ in range(25):
			out = self.conv2(self.relu(self.conv1(self.relu(out))))
			out = torch.add(out, inputs)

		out = self.output(self.relu(out))
		out = torch.add(out, residual)
		return out
        
 # vdsr
class Conv_ReLU_Block(nn.Module):

    def __init__(self):

        super(Conv_ReLU_Block, self).__init__()

        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        

    def forward(self, x):

        return self.relu(self.conv(x))

        

class EDSR(nn.Module):

    def __init__(self, in_channels, out_channels, num_features, num_blocks, res_scale, upscale_factor):

        super(EDSR, self).__init__()

        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)

        self.input = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.output = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels

                m.weight.data.normal_(0, sqrt(2. / n))

                

    def make_layer(self, block, num_of_layer):

        layers = []

        for _ in range(num_of_layer):

            layers.append(block())

        return nn.Sequential(*layers)



    def forward(self, x):

        residual = x

        out = self.relu(self.input(x))

        out = self.residual_layer(out)

        out = self.output(out)

        out = torch.add(out,residual)

        return out