import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#THIS CODE IS NOT FULLY OPERATIONAL
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
import sys
sys.path.append('../')
from physics import *
from Adam import Adam
import torch
import torch.nn as nn
if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 16 # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(4, self.width)  
        
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.w4=nn.Conv2d(self.width,self.width,1)

        self.fc1 = nn.Linear(self.width, 256)
        #self.fc2 = nn.Linear(128, 3)
        self.fc_u = nn.Linear(256, 1)
        self.fc_v = nn.Linear(256, 1)
        self.fc_p = nn.Linear(256, 1)
        self.fc_T = nn.Linear(256, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        grid.requires_grad=True
        
        #print(x.shape,grid.shape)
        x1 = torch.cat((x, grid), dim=-1)
       #print(x.shape)
        x2 = self.fc0(x1)
        x3 = x2.permute(0, 3, 1, 2)
        #print(x3.requires_grad)
        
        x4 = F.pad(x3, [0, self.padding, 0, self.padding])
        #print(x.shape)
        x5 = self.conv0(x4)
        x6 = self.w0(x4)
        x7 = x5 + x6
        x8 = F.gelu(x7)

        x9 = self.conv1(x8)
        x10 = self.w1(x8)
        x11 = x9 + x10
        x12 = F.gelu(x11)

        x13 = self.conv2(x12)
        x14 = self.w2(x12)
        x15 = x13 + x14
        x16 = F.gelu(x15)

        x17 = self.conv3(x16)
        x18 = self.w3(x16)
        x19 = x17 + x18
        x19=F.gelu(x19)
        xnew=self.conv4(x19)
        xnew2=self.w4(x19)
        xnew3=xnew+xnew2

        x20 = xnew3[..., :-self.padding, :-self.padding]
        x21 = x20.permute(0, 2, 3, 1)
        x22 = self.fc1(x21)
        x23 = F.gelu(x22)
        #x = self.fc2(x)
        #print(x.device)
        u = self.fc_u(x23)
        v = self.fc_v(x23)
        p = self.fc_p(x23)
        T = self.fc_T(x23)
        #print(x.is_leaf)
        
        return u,v,p,T

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
