
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


################################################################
# configs
################################################################





ntrain = 1000
ntest = 200
N = 1200

batch_size = 5
learning_rate = 0.001

epochs = 501
step_size =5
gamma = 0.5

modes = 12
width = 24

r1 = 1
r2 = 1
s1 = int(((100 - 1) / r1) + 1)
s2 = int(((100 - 1) / r2) + 1)

inputX = np.load('/home/iyer.ris/inputx.npy')
#inputX = torch.tensor(inputX, dtype=torch.float)
inputY = np.load('/home/iyer.ris/inputy.npy')
diameter = 1.0  # pipe diameter
radius = diameter / 2.0  # pipe radius
num_points = 100  # number of grid points
# Generate evenly spaced points in the Y dimension
y_values = np.linspace(-radius, radius, num_points)
x_values=np.linspace(0,10,num_points)
# Repeat these points in the X dimension
#inputY = torch.tensor(inputY, dtype=torch.float)
#Y_data1 = np.tile(y_values, (num_points, 1))
Y_data1 = np.repeat(inputY[np.newaxis, :, :], 2310, axis=0)
inputX=torch.tensor(np.repeat(inputX[np.newaxis, :, :], 2310, axis=0))
inputY = torch.tensor(Y_data1, dtype=torch.float)


input = torch.stack([inputX, inputY], dim=-1)

output1 = np.load('/home/iyer.ris/outputv.npy')
output1 = torch.tensor(np.repeat(output1[np.newaxis, :, :], 2310, axis=0))
output2 = np.load('/home/iyer.ris/outputp.npy')
output2 = torch.tensor(np.repeat(output2[np.newaxis, :, :], 2310, axis=0))
output2=torch.unsqueeze(output2,axis=-1)
output3 = np.load('/home/iyer.ris/outputT.npy')
output3 = torch.tensor(np.repeat(output3[np.newaxis, :, :], 2310, axis=0))
output3=torch.unsqueeze(output3,axis=-1)
print("Shape of output1:", output1.shape)
print("Shape of output2:", output2.shape)
print("Shape of output3:", output3.shape)
output_stacked = torch.stack([output2, output3], dim=3).squeeze()
print(output_stacked.shape)
output = torch.cat([output1, output_stacked], dim=3)
print(output.shape,input.shape)

x_train = input[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]
y_train = output[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]
x_test = input[:N][-ntest:, ::r1, ::r2][:, :s1, :s2]
y_test = output[:N][-ntest:, ::r1, ::r2][:, :s1, :s2]
x_train = x_train.reshape(ntrain, s1, s2, 2)
x_test = x_test.reshape(ntest, s1, s2, 2)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          shuffle=False)
test_loader2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1,
                                          shuffle=False)

model = FNO2d(modes, modes, width).to(device)
print(count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
#physics_loss = PhysicsLossFFT(model).to(device)
#physics_loss=PhysicsLoss(model).to(device)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x.requires_grad = True
        
        optimizer.zero_grad()
        u, v, p, T = model(x)
        
        # Concatenate the model outputs along the last dimension
        output = torch.cat((u, v, p, T), dim=-1)
        
        loss = myloss(output.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()
        
        optimizer.step()
        train_l2 += loss.item()
    
    scheduler.step()
    
    model.eval()
    test_l2 = 0.0
    
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        x.requires_grad = True
        u, v, p, T = model(x)
        
        # Concatenate the model outputs along the last dimension
        output = torch.cat((u, v, p, T), dim=-1)
        
        test_l2 += myloss(output.view(batch_size, -1), y.view(batch_size, -1)).item()
    
    train_l2 /= ntrain
    test_l2 /= ntest
    
    t2 = default_timer()
    print(ep, t2 - t1, train_l2, test_l2)
    
    if ep % step_size == 0:
        torch.save(model, '/home/iyer.ris/pipe/pipemodelrun_' + str(ep))


