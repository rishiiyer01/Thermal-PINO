import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn
import torch.nn.functional as F
import operator
from functools import reduce
#################################################
#
# physics losses contained in this file
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def find_tensor_in_graph(tensor_to_find, grad_fn):
    if grad_fn is None:
        return False
    
    for next_fn, _ in grad_fn.next_functions:
        # Check if next_fn is actually a tuple (it should be)
        if isinstance(next_fn, tuple):
            # Take the first element from the tuple (the grad_fn object)
            next_fn = next_fn[0]
        
        if next_fn is None:
            continue
        
        # Add a check for saved_tensors
        if hasattr(next_fn, 'saved_tensors'):
            for saved_tensor in next_fn.saved_tensors:
                if id(saved_tensor) == id(tensor_to_find):
                    return True
        
        # Recursion into the next functions
        if find_tensor_in_graph(tensor_to_find, next_fn):
            return True

    return False
class PhysicsLossFFT(nn.Module):
    def __init__(self,model, alpha=(0.143*10**-6), beta=(1/1000), nu=0.000001):
        super(PhysicsLossFFT, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.model = model
    def forward(self,x):
        y = x[:, :, :, 1].unsqueeze(-1)
        xsliced = x[:, :, :, 0].unsqueeze(-1)
        xnew = torch.cat((xsliced, y), dim=-1)
        
        #print('xnew',xnew.shape)
        u,v,p,T=self.model(xnew)
        #print(u.shape)
        nx=u.size(1)
        ny=u.size(2)
        dx = 10 / (nx)
        dy=1/ny
        #wavenumbers in x direction, wave numbers in y direction for each of the 4 quantities
        upad = F.pad(u, [0, 0, 0, 16, 0, 16, 0, 0])
        vpad = F.pad(v, [0, 0, 0, 16, 0, 16, 0, 0])
        ppad = F.pad(p, [0, 0, 0, 16, 0, 16, 0, 0])
        Tpad = F.pad(T, [0, 0, 0, 16, 0, 16, 0, 0])
        u_wave=torch.fft.fft2(upad,dim=[1,2])
        v_wave=torch.fft.fft2(vpad,dim=[1,2])
        p_wave=torch.fft.fft2(ppad,dim=[1,2])
        T_wave=torch.fft.fft2(Tpad,dim=[1,2])
        #print(u_wave.shape)
        k_x = torch.fft.fftfreq(nx+16, d=dx) * 2 * np.pi # 2 * pi
        k_y = torch.fft.fftfreq(ny+16, d=dy) * 2 * np.pi  # 2 * pi
        k_x = k_x.to(u_wave.device)
        k_y = k_y.to(u_wave.device)
        k_x = k_x.view(1, 129+16, 1, 1)
        k_y = k_y.view(1, 1, 129+16, 1)
        u_x_wave = 2j * np.pi * k_x * u_wave
        u_xx_wave = -(2 * np.pi * k_x)**2 * u_wave
        u_y_wave = 2j * np.pi * k_y * u_wave
        u_yy_wave = -(2 * np.pi * k_y)**2 * u_wave

        v_x_wave = 2j * np.pi * k_x * v_wave
        v_xx_wave = -(2 * np.pi * k_x)**2 * v_wave
        v_y_wave = 2j * np.pi * k_y * v_wave
        v_yy_wave = -(2 * np.pi * k_y)**2 * v_wave

        p_x_wave=2j*np.pi*k_x*p_wave
        p_y_wave=2j*np.pi*k_y*p_wave

        T_x_wave=2j*np.pi*k_x*T_wave
        T_xx_wave=-(2*np.pi*k_x)**2*T_wave
        T_y_wave=2j*np.pi*k_y*T_wave
        T_yy_wave=-(2*np.pi*k_y)**2*T_wave
        # Perform inverse FFT to get derivatives in spatial domain
        u_x = torch.fft.irfft2(u_x_wave, s=[nx+16, ny+16], dim=[1, 2])
   
        u_x=u_x[:, :129, :129, :]
     
        u_xx = torch.fft.irfft2(u_xx_wave, s=[nx+16, ny+16], dim=[1, 2])
        u_xx=u_xx[:, :129, :129, :]
        u_y = torch.fft.irfft2(u_y_wave, s=[nx+16, ny+16], dim=[1, 2])
        u_y=u_y[:, :129, :129, :]
        u_yy = torch.fft.irfft2(u_yy_wave, s=[nx+16, ny+16], dim=[1, 2])
        u_yy=u_yy[:, :129, :129, :]

        v_x = torch.fft.irfft2(v_x_wave, s=[nx+16, ny+16], dim=[1, 2])
        v_x=v_x[:, :129, :129, :]
        v_xx = torch.fft.irfft2(v_xx_wave, s=[nx+16, ny+16], dim=[1, 2])
        v_xx=v_xx[:, :129, :129, :]
        v_y = torch.fft.irfft2(v_y_wave, s=[nx+16, ny+16], dim=[1, 2])
        v_y=v_y[:, :129, :129, :]
        v_yy = torch.fft.irfft2(v_yy_wave, s=[nx+16, ny+16], dim=[1, 2])
        v_yy=v_yy[:, :129, :129, :]

        p_x=torch.fft.irfft2(p_x_wave,s=[nx+16, ny+16],dim=[1,2])
        p_x=p_x[:, :129, :129, :]
        p_y=torch.fft.irfft2(p_y_wave,s=[nx+16, ny+16],dim=[1,2])
        p_y=p_y[:, :129, :129, :]

        T_x=torch.fft.irfft2(T_x_wave,s=[nx+16, ny+16],dim=[1,2])
        T_x=T_x[:, :129, :129, :]
        T_xx=torch.fft.irfft2(T_xx_wave,s=[nx+16, ny+16], dim=[1, 2])
        T_xx=T_xx[:, :129, :129, :]
        T_y=torch.fft.irfft2(T_y_wave,s=[nx+16, ny+16],dim=[1,2])
        T_y=T_y[:, :129, :129, :]
        T_yy=torch.fft.irfft2(T_yy_wave,s=[nx+16, ny+16],dim=[1,2])
        T_yy=T_yy[:, :129, :129, :]
        
        #print(u_wave.shape,u_x_wave.shape,k_x.shape)
        # Continuity Loss
        u_internal=u[:, 1:-1, 1:-1, :]
        u_x_internal = u_x[:, 1:-1, 1:-1, :]
        u_y_internal = u_y[:, 1:-1, 1:-1, :]
        u_xx_internal=u_xx[:, 1:-1, 1:-1, :]
        u_yy_internal=u_yy[:, 1:-1, 1:-1, :]

        v_internal=v[:, 1:-1, 1:-1, :]
        v_x_internal=v_x[:, 1:-1, 1:-1, :]
        v_y_internal=v_y[:, 1:-1, 1:-1, :]
        v_xx_internal=v_xx[:, 1:-1, 1:-1, :]
        v_yy_internal=v_yy[:, 1:-1, 1:-1, :]

        T_x_internal=T_x[:, 1:-1, 1:-1, :]
        T_y_internal=T_y[:, 1:-1, 1:-1, :]
        T_xx_internal=T_xx[:, 1:-1, 1:-1, :]
        T_yy_internal=T_yy[:, 1:-1, 1:-1, :]

        p_x_internal=p_x[:, 1:-1, 1:-1, :]
        p_y_internal=p_y[:, 1:-1, 1:-1, :]

        L_cont1 = (u_x_internal + v_y_internal)
        f=torch.zeros(u_x_internal.shape,device=u.device)
        L_cont=F.mse_loss(L_cont1,f)
        # Momentum Loss (u component and v component)
        L_mom_u1 = (u_internal*u_x_internal + v_internal*u_y_internal + self.beta*p_x_internal - self.nu*(u_xx_internal + u_yy_internal))
        L_mom_u=F.mse_loss(L_mom_u1,f)
        L_mom_v1 = (u_internal*v_x_internal + v_internal*v_y_internal + self.beta*p_y_internal - self.nu*(v_xx_internal + v_yy_internal))
        L_mom_v=F.mse_loss(L_mom_v1,f)
        #Heat Advection Equation Loss
        L_heat1= (u_internal*T_x_internal + v_internal*T_y_internal - self.alpha*(T_xx_internal + T_yy_internal))
        L_heat=F.mse_loss(L_heat1,f)
        u_avg=5
        u_inlet_expected = u_avg*1.5*(1 - 4*((y[:, 0,:, :] ).pow(2)))
        T_wall=350
        T_center=293
        T_inlet_expected = T_wall + (T_center - T_wall) * (1 - 4*((y[:, 0,:, :] ).pow(2)))

        #Loss due to boundary conditions
        #Loss due to boundary conditions
        left_T_bc1 = T[:, 0, :, :]
        left_T_bc=F.mse_loss(left_T_bc1,T_inlet_expected)
        #right_T_bc = T[:, :, -1, :].pow(2).sum()
        top_T_bc1 = T[:, :, 0, :]
        T_wallvec=T_wall*torch.ones_like(top_T_bc1,device=u.device)
        top_T_bc=F.mse_loss(top_T_bc1,T_wallvec)
        bottom_T_bc1 = T[:,:, -1, :]
        bottom_T_bc=F.mse_loss(bottom_T_bc1,T_wallvec)
        #print(T_wallvec)

        boundaryzero=torch.zeros(u[:, :, 0, :].shape,device=u.device)

        left_u_bc = F.mse_loss(u[:, 0, :, :],u_inlet_expected)
        #print(u_inlet_expected)
        #right_u_bc = u[:, :, -1, :].pow(2).sum()
        top_u_bc = F.mse_loss(u[:, :, 0, :],boundaryzero)
        bottom_u_bc = F.mse_loss(u[:, :, -1, :],boundaryzero)

        vzeroleft=torch.zeros(v[:, 0,:, :].shape,device=u.device)
        left_v_bc = F.mse_loss(v[:, 0,:, :],vzeroleft)
        #right_v_bc = v[:, :, -1, :].pow(2).sum()
        top_v_bc = F.mse_loss(v[:, :,0, :],boundaryzero)
        bottom_v_bc = F.mse_loss(v[:, :,-1, :],boundaryzero)

        #left_p_bc = p[:, :, 0, :].pow(2).sum()
        right_p_bc = F.mse_loss(p[:, -1,:, :],vzeroleft)
        #top_p_bc = p[:, 0, :, :].pow(2).sum()
        #bottom_p_bc = p[:, -1, :, :].pow(2).sum()
        bc_loss=torch.norm(F.softmax(2*left_T_bc+top_T_bc+bottom_T_bc+2*left_u_bc+top_u_bc+bottom_u_bc+2*left_v_bc+top_v_bc+bottom_v_bc+right_p_bc))
        ###integral control volume soft constraints
        #first temperature: 
        #inletTemp=(u[:,0,:,:]*T[:,0,:,:]).sum()
        #outletTemp=(u[:,-1,:,:]*T[:,-1,:,:]).sum()
        #TempAdvection=inletTemp-outletTemp
        #TempAdvectionLoss=F.mse_loss(TempAdvection,torch.zeros_like(TempAdvection))
        #velocity only in x direction
        #inletU=(u[:,0,:,:]*u[:,0,:,:]).sum()
        #outletU=(u[:,-1,:,:]*u[:,-1,:,:]).sum()
       #expected=(u_inlet_expected*u_inlet_expected).sum()
        
        
        #flowLoss1=F.mse_loss(inletU,expected)
        #flowLoss2=F.mse_loss(outletU,expected)
        #integral_loss=torch.norm(flowLoss2)
        # Total Loss
        L = torch.norm(L_mom_u + L_mom_v+2*L_cont+L_heat+bc_loss)
        #print(L)
        return L


class PhysicsLossSIM(nn.Module):
    def __init__(self,model, alpha=(0.143*10**-6), beta=(1/1000), nu=0.000001):
        super(PhysicsLossSIM, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.model = model
    def forward(self,x):
        y = x[:, :, :, 1].unsqueeze(-1)
        xsliced = x[:, :, :, 0].unsqueeze(-1)
        xnew = torch.cat((xsliced, y), dim=-1)
        
        #print('xnew',xnew.shape)
        u,v,p,T=self.model(xnew)
        nx=u.size(1)
        ny=u.size(2)
        dx = 10 / (nx)
        dy=1/ny
        #print(grid.shape)
        #print("after slicing",x.grad_fn)

        
        T_x=torch.autograd.grad(T, xsliced,grad_outputs=torch.ones_like(T),retain_graph=True, create_graph=True)[0]
        T_y=torch.autograd.grad(T, y,grad_outputs=torch.ones_like(T),retain_graph=True, create_graph=True)[0]
        
        T_xx= torch.autograd.grad(T_x,xsliced,grad_outputs=torch.ones_like(T),retain_graph=True, create_graph=True)[0]
        T_yy= torch.autograd.grad(T_y,y,grad_outputs=torch.ones_like(T),retain_graph=True, create_graph=True)[0]
        #print(u_wave.shape,u_x_wave.shape,k_x.shape)
        # Continuity Loss
        

        #L_cont1 = (u_x + v_y)
        
        f=torch.zeros(u_x.shape,device=u.device)
        
        # Momentum Loss (u component and v component)
        
        #pressure
        #(u[2:, 1:-1] - u[0:-2, 1:-1])
        
        #
        
        
        #Heat Advection Equation Loss
        L_heat1= (u*T_x + v*T_y - self.alpha*(T_xx + T_yy))
        L_heat=F.mse_loss(L_heat1,f)
        u_avg=5
        u_expected = u_avg*1.5*(1 - 4*((y ).pow(2)))
        L_u=F.mse_loss(u,u_expected)
        L_v=F.mse_loss(v,torch.zeros_like(u_expected))
        T_wall=350
        T_center=293
        T_inlet_expected = T_wall + (T_center - T_wall) * (1 - 4*((y[:, 0,:, :] ).pow(2)))

        #Loss due to boundary conditions
        #Loss due to boundary conditions
        left_T_bc1 = T[:, 0, :, :]
        left_T_bc=F.mse_loss(left_T_bc1,T_inlet_expected)
        #right_T_bc = T[:, :, -1, :].pow(2).sum()
        top_T_bc1 = T[:, :, 0, :]
        T_wallvec=T_wall*torch.ones_like(top_T_bc1,device=u.device)
        top_T_bc=F.mse_loss(top_T_bc1,T_wallvec)
        bottom_T_bc1 = T[:,:, -1, :]
        bottom_T_bc=F.mse_loss(bottom_T_bc1,T_wallvec)
        #print(T_wallvec)

        boundaryzero=torch.zeros_like(u[:, :, 0, :])
        vzeroleft=torch.zeros(v[:, 0,:, :].shape,device=u.device)
        left_u_bc = F.mse_loss(u[:, 0, :, :],u_inlet_expected)
        #print(u_inlet_expected)
        right_u_bc = F.mse_loss(u_x[:, -1,:, :],vzeroleft)
        top_u_bc = F.mse_loss(u[:, :, 0, :],boundaryzero)
        bottom_u_bc = F.mse_loss(u[:, :, -1, :],boundaryzero)

    
        left_v_bc = F.mse_loss(v[:, 0,:, :],vzeroleft)
        right_v_bc = F.mse_loss(v_x[:, -1,:, :],vzeroleft)
        top_v_bc = F.mse_loss(v[:, :,0, :],torch.zeros_like(v[:,:,0,:]))
        bottom_v_bc = F.mse_loss(v[:, :,-1, :],torch.zeros_like(v[:,:,0,:]))

        #left_p_bc = p[:, :, 0, :].pow(2).sum()
        right_p_bc = F.mse_loss(p[:, -1,:, :],vzeroleft)
        top_p_bc = F.mse_loss(p_y[:, :, 0, :],boundaryzero)
        bottom_p_bc = F.mse_loss(p_y[:, :, -1, :],boundaryzero)
        bc_loss=torch.norm(left_T_bc+top_T_bc+bottom_T_bc+1.5*left_u_bc+1.5*top_u_bc+1.5*bottom_u_bc+1.5*right_u_bc+1.5*right_v_bc+1.5*left_v_bc+5*top_v_bc+5*bottom_v_bc+right_p_bc+top_p_bc+bottom_p_bc)
        ###integral control volume soft constraints
        #first temperature: 
        #inletTemp=(u[:,0,:,:]*T[:,0,:,:]).sum()
        #outletTemp=(u[:,-1,:,:]*T[:,-1,:,:]).sum()
        #TempAdvection=inletTemp-outletTemp
        #TempAdvectionLoss=F.mse_loss(TempAdvection,torch.zeros_like(TempAdvection))
        #velocity only in x direction
        #inletU=(u[:,0,:,:]*u[:,0,:,:]).sum()
        #outletU=(u[:,-1,:,:]*u[:,-1,:,:]).sum()
       #expected=(u_inlet_expected*u_inlet_expected).sum()
        
        
        #flowLoss1=F.mse_loss(inletU,expected)
        #flowLoss2=F.mse_loss(outletU,expected)
        #integral_loss=torch.norm(flowLoss2)
        # Total Loss
        v_loss=F.mse_loss(v,torch.zeros_like(v))
        L = torch.norm(L_mom_u**2 + L_mom_v**2 +L_cont**2 +L_heat**2+2*bc_loss**2+2*v_loss**2)
        
        #print(L)
        return L
class PhysicsLoss(nn.Module):
    def __init__(self,model, alpha=(0.143*10**-6), beta=(1/1000), nu=0.000001):
        super(PhysicsLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.model = model
    def forward(self,x):
        # Calculate the gradients needed
        #for steady state
        
       
                

        

        # Assuming x and u are your tensors
        #print(x.shape)

        
        
        
        y = x[:, :, :, 1].unsqueeze(-1)
        xsliced = x[:, :, :, 0].unsqueeze(-1)
        xnew = torch.cat((xsliced, y), dim=-1)
        
        #print('xnew',xnew.shape)
        u,v,p,T=self.model(xnew)
        nx=u.size(1)
        ny=u.size(2)
        dx = 10 / (nx)
        dy=1/ny
        #print(grid.shape)
        #print("after slicing",x.grad_fn)

        u_x = torch.autograd.grad(u, xsliced,grad_outputs=torch.ones_like(u),retain_graph=True, create_graph=True)[0]
        #print(u_x)
        u_y = torch.autograd.grad(u, y,grad_outputs=torch.ones_like(u),retain_graph=True, create_graph=True)[0]
        T_x=torch.autograd.grad(T, xsliced,grad_outputs=torch.ones_like(T),retain_graph=True, create_graph=True)[0]
        T_y=torch.autograd.grad(T, y,grad_outputs=torch.ones_like(T),retain_graph=True, create_graph=True)[0]
        v_x = torch.autograd.grad(v, xsliced,grad_outputs=torch.ones_like(v),retain_graph=True, create_graph=True)[0]
        v_y = torch.autograd.grad(v, y,grad_outputs=torch.ones_like(v),retain_graph=True, create_graph=True)[0]
        
        p_x = torch.autograd.grad(p, xsliced,grad_outputs=torch.ones_like(p),retain_graph=True, create_graph=True)[0]
        p_y = torch.autograd.grad(p, y,grad_outputs=torch.ones_like(p),retain_graph=True, create_graph=True)[0]
        p_xx=torch.autograd.grad(p_x, xsliced,grad_outputs=torch.ones_like(p),retain_graph=True, create_graph=True)[0]
        p_yy=torch.autograd.grad(p_y, y,grad_outputs=torch.ones_like(p),retain_graph=True, create_graph=True)[0]
        u_xx= torch.autograd.grad(u_x,xsliced,grad_outputs=torch.ones_like(u),retain_graph=True, create_graph=True)[0]
        u_yy= torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(u),retain_graph=True, create_graph=True)[0]
        v_xx= torch.autograd.grad(v_x,xsliced,grad_outputs=torch.ones_like(v),retain_graph=True, create_graph=True)[0]
        v_yy= torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(v), retain_graph=True,create_graph=True)[0]
        T_xx= torch.autograd.grad(T_x,xsliced,grad_outputs=torch.ones_like(T),retain_graph=True, create_graph=True)[0]
        T_yy= torch.autograd.grad(T_y,y,grad_outputs=torch.ones_like(T),retain_graph=True, create_graph=True)[0]
        #print(u_wave.shape,u_x_wave.shape,k_x.shape)
        # Continuity Loss
        

        #L_cont1 = (u_x + v_y)
        #pressure poisson equation
        L_cont1=self.beta*(p_xx+p_yy)+(u_x*u_x+2*u_y*v_x+v_y*v_y)
        f=torch.zeros(u_x.shape,device=u.device)
        L_cont=F.mse_loss(L_cont1,f)
        # Momentum Loss (u component and v component)
        L_mom_u1 = (u*u_x + v*u_y + self.beta*p_x - self.nu*(u_xx + u_yy))
        #pressure
        #(u[2:, 1:-1] - u[0:-2, 1:-1])
        
        #
        L_mom_u=F.mse_loss(L_mom_u1,f)
        L_mom_v1 = (u*v_x + v*v_y + self.beta*p_y- self.nu*(v_xx + v_yy))
        L_mom_v=F.mse_loss(L_mom_v1,f)
        #Heat Advection Equation Loss
        L_heat1= (u*T_x + v*T_y - self.alpha*(T_xx + T_yy))
        L_heat=F.mse_loss(L_heat1,f)
        u_avg=5
        u_inlet_expected = u_avg*1.5*(1 - 4*((y[:, 0,:, :] ).pow(2)))
        T_wall=350
        T_center=293
        T_inlet_expected = T_wall + (T_center - T_wall) * (1 - 4*((y[:, 0,:, :] ).pow(2)))

        #Loss due to boundary conditions
        #Loss due to boundary conditions
        left_T_bc1 = T[:, 0, :, :]
        left_T_bc=F.mse_loss(left_T_bc1,T_inlet_expected)
        #right_T_bc = T[:, :, -1, :].pow(2).sum()
        top_T_bc1 = T[:, :, 0, :]
        T_wallvec=T_wall*torch.ones_like(top_T_bc1,device=u.device)
        top_T_bc=F.mse_loss(top_T_bc1,T_wallvec)
        bottom_T_bc1 = T[:,:, -1, :]
        bottom_T_bc=F.mse_loss(bottom_T_bc1,T_wallvec)
        #print(T_wallvec)

        boundaryzero=torch.zeros_like(u[:, :, 0, :])
        vzeroleft=torch.zeros(v[:, 0,:, :].shape,device=u.device)
        left_u_bc = F.mse_loss(u[:, 0, :, :],u_inlet_expected)
        #print(u_inlet_expected)
        right_u_bc = F.mse_loss(u_x[:, -1,:, :],vzeroleft)
        top_u_bc = F.mse_loss(u[:, :, 0, :],boundaryzero)
        bottom_u_bc = F.mse_loss(u[:, :, -1, :],boundaryzero)

    
        left_v_bc = F.mse_loss(v[:, 0,:, :],vzeroleft)
        right_v_bc = F.mse_loss(v_x[:, -1,:, :],vzeroleft)
        top_v_bc = F.mse_loss(v[:, :,0, :],torch.zeros_like(v[:,:,0,:]))
        bottom_v_bc = F.mse_loss(v[:, :,-1, :],torch.zeros_like(v[:,:,0,:]))

        #left_p_bc = p[:, :, 0, :].pow(2).sum()
        right_p_bc = F.mse_loss(p[:, -1,:, :],vzeroleft)
        top_p_bc = F.mse_loss(p_y[:, :, 0, :],boundaryzero)
        bottom_p_bc = F.mse_loss(p_y[:, :, -1, :],boundaryzero)
        bc_loss=torch.norm(left_T_bc+top_T_bc+bottom_T_bc+1.5*left_u_bc+1.5*top_u_bc+1.5*bottom_u_bc+1.5*right_u_bc+1.5*right_v_bc+1.5*left_v_bc+5*top_v_bc+5*bottom_v_bc+right_p_bc+top_p_bc+bottom_p_bc)
        ###integral control volume soft constraints
        #first temperature: 
        #inletTemp=(u[:,0,:,:]*T[:,0,:,:]).sum()
        #outletTemp=(u[:,-1,:,:]*T[:,-1,:,:]).sum()
        #TempAdvection=inletTemp-outletTemp
        #TempAdvectionLoss=F.mse_loss(TempAdvection,torch.zeros_like(TempAdvection))
        #velocity only in x direction
        #inletU=(u[:,0,:,:]*u[:,0,:,:]).sum()
        #outletU=(u[:,-1,:,:]*u[:,-1,:,:]).sum()
       #expected=(u_inlet_expected*u_inlet_expected).sum()
        
        
        #flowLoss1=F.mse_loss(inletU,expected)
        #flowLoss2=F.mse_loss(outletU,expected)
        #integral_loss=torch.norm(flowLoss2)
        # Total Loss
        v_loss=F.mse_loss(v,torch.zeros_like(v))
        L = torch.norm(L_mom_u**2 + L_mom_v**2 +L_cont**2 +L_heat**2+2*bc_loss**2+2*v_loss**2)
        
        #print(L)
        return L
class physicsLossPaper(nn.Module):
    def __init__(self,model, alpha=(0.143*10**-6), beta=(1/1000), nu=0.000001):
        super(physicsLossPaper, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.model = model
    def forward(self,x):
        # Calculate the gradients needed
        #for steady state
        
       
                

        

        # Assuming x and u are your tensors
        #print(x.shape)

        
        
        
        y = x[:, :, :, 1].unsqueeze(-1)
        xsliced = x[:, :, :, 0].unsqueeze(-1)
        xnew = torch.cat((xsliced, y), dim=-1)
        
        #print('xnew',xnew.shape)
        u,v,p,T=self.model(xnew)
        nx=u.size(1)
        ny=u.size(2)
        dx = 10 / (nx)
        dy=1/ny
       # Central difference for interior points for u_x
        u_x_internal = (u[:, 2:, 1:-1,:] - u[:, :-2, 1:-1,:]) / (2 * dx)
        u_y_internal= (u[:, 1:-1,2:,:] - u[:, 1:-1,:-2,:]) / (2 * dy)
        u_xx_internal= (u[:, 2:, 1:-1,:] - 2 * u[:, 1:-1, 1:-1,:] + u[:, :-2, 1:-1,:]) / (dx ** 2)
        u_yy_internal=(u[:, 1:-1,2:,:] - 2 * u[:, 1:-1, 1:-1,:] + u[:, 1:-1,:-2,:]) / (dy ** 2)
        
        v_x_internal = (v[:, 2:, 1:-1,:] - v[:, :-2, 1:-1,:]) / (2 * dx)
        v_y_internal= (v[:, 1:-1,2:,:] - v[:, 1:-1,:-2,:]) / (2 * dy)
        v_xx_internal= (v[:, 2:, 1:-1,:] - 2 * v[:, 1:-1, 1:-1,:] + v[:, :-2, 1:-1,:]) / (dx ** 2)
        v_yy_internal=(v[:, 1:-1,2:,:] - 2 * v[:, 1:-1, 1:-1,:] + v[:, 1:-1,:-2,:]) / (dy ** 2)
        
        T_x_internal = (T[:, 2:, 1:-1,:] - T[:, :-2, 1:-1,:]) / (2 * dx)
        T_y_internal= (T[:, 1:-1,2:,:] - T[:, 1:-1,:-2,:]) / (2 * dy)
        T_xx_internal= (T[:, 2:, 1:-1,:] - 2 * T[:, 1:-1, 1:-1,:] + T[:, :-2, 1:-1,:]) / (dx ** 2)
        T_yy_internal=(T[:, 1:-1,2:,:] - 2 * T[:, 1:-1, 1:-1,:] + T[:, 1:-1,:-2,:]) / (dy ** 2)

        p_x_internal=(p[:, 2:, 1:-1,:] - p[:, :-2, 1:-1,:]) / (2 * dx)
        p_y_internal=(p[:, 1:-1,2:,:] - p[:, 1:-1,:-2,:]) / (2 * dy)
        # Continuity Loss
        u_internal=u[:, 1:-1, 1:-1, :]
        

        v_internal=v[:, 1:-1, 1:-1, :]


        
        

        

        L_cont1 = (u_x_internal + v_y_internal)
        f=torch.zeros(u_x_internal.shape,device=u.device)
        L_cont=F.mse_loss(L_cont1,f)
        # Momentum Loss (u component and v component)
        L_mom_u1 = (u_internal*u_x_internal + v_internal*u_y_internal + self.beta*p_x_internal - self.nu*(u_xx_internal + u_yy_internal))
        L_mom_u=F.mse_loss(L_mom_u1,f)
        L_mom_v1 = (u_internal*v_x_internal + v_internal*v_y_internal + self.beta*p_y_internal - self.nu*(v_xx_internal + v_yy_internal))
        L_mom_v=F.mse_loss(L_mom_v1,f)
        #Heat Advection Equation Loss
        L_heat1= (u_internal*T_x_internal + v_internal*T_y_internal - self.alpha*(T_xx_internal + T_yy_internal))
        L_heat=F.mse_loss(L_heat1,f)
        u_avg=10**-3
        u_inlet_expected = u_avg*1.5*(1 - 4*((y[:, 0,:, :] ).pow(2)))
        T_wall=350
        T_center=293
        T_inlet_expected = T_wall + (T_center - T_wall) * (1 - 4*((y[:, 0,:, :] ).pow(2)))

        #Loss due to boundary conditions
        #Loss due to boundary conditions
        left_T_bc1 = T[:, 0, :, :]
        left_T_bc=F.mse_loss(left_T_bc1,T_inlet_expected)
        #right_T_bc = T[:, :, -1, :].pow(2).sum()
        top_T_bc1 = T[:, :, 0, :]
        T_wallvec=T_wall*torch.ones_like(top_T_bc1,device=u.device)
        top_T_bc=F.mse_loss(top_T_bc1,T_wallvec)
        bottom_T_bc1 = T[:,:, -1, :]
        bottom_T_bc=F.mse_loss(bottom_T_bc1,T_wallvec)
        #print(T_wallvec)

        boundaryzero=torch.zeros(u[:, :, 0, :].shape,device=u.device)
        vzeroleft=torch.zeros(v[:, 0,:, :].shape,device=u.device)
        left_u_bc = F.mse_loss(u[:, 0, :, :],u_inlet_expected)
        #print(u_inlet_expected)
        right_u_bc = F.mse_loss(u_x[:, -1,:, :],vzeroleft)
        top_u_bc = F.mse_loss(u[:, :, 0, :],boundaryzero)
        bottom_u_bc = F.mse_loss(u[:, :, -1, :],boundaryzero)

        
        left_v_bc = F.mse_loss(v[:, 0,:, :],vzeroleft)
        right_v_bc = F.mse_loss(v_x[:, -1,:, :],vzeroleft)
        top_v_bc = F.mse_loss(v[:, :,0, :],boundaryzero)
        bottom_v_bc = F.mse_loss(v[:, :,-1, :],boundaryzero)

        #left_p_bc = p[:, :, 0, :].pow(2).sum()
        right_p_bc = F.mse_loss(p[:, -1,:, :],vzeroleft)
        top_p_bc = F.mse_loss(p_y[:, 0, :, :],boundaryzero)
        bottom_p_bc = F.mse_loss(p_y[:, -1, :, :],boundaryzero)
        bc_loss=torch.norm(left_T_bc+top_T_bc+bottom_T_bc+left_u_bc+top_u_bc+bottom_u_bc+right_u_bc+right_v_bc+left_v_bc+top_v_bc+bottom_v_bc+right_p_bc+top_p_bc+bottom_p_bc)
        ###integral control volume soft constraints
        #first temperature: 
        #inletTemp=(u[:,0,:,:]*T[:,0,:,:]).sum()
        #outletTemp=(u[:,-1,:,:]*T[:,-1,:,:]).sum()
        #TempAdvection=inletTemp-outletTemp
        #TempAdvectionLoss=F.mse_loss(TempAdvection,torch.zeros_like(TempAdvection))
        #velocity only in x direction
        #inletU=(u[:,0,:,:]*u[:,0,:,:]).sum()
        #outletU=(u[:,-1,:,:]*u[:,-1,:,:]).sum()
       #expected=(u_inlet_expected*u_inlet_expected).sum()
        
        
        #flowLoss1=F.mse_loss(inletU,expected)
        #flowLoss2=F.mse_loss(outletU,expected)
        #integral_loss=torch.norm(flowLoss2)
        # Total Loss
        L = torch.norm(L_mom_u + L_mom_v+L_cont+L_heat+bc_loss)
        
        #print(L)
        return L
#class boundaryLoss(nn.module):
 #   def __init__(self,model, alpha=(0.143*10**-6), beta=(1/1000), nu=0.000001):
 #       super(physicsLossFVM, self).__init__()
 #       
 #       self.model = model
 #   def forward(self,mask,x,y):
        
        


class physicsLossFVM(nn.Module):
    def __init__(self,model, alpha=(0.143*10**-6), beta=(1/1000), nu=0.000001):
        super(physicsLossFVM, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.model = model
    def forward(self,x):
        # Calculate the gradients needed
        #for steady state
        
       
                

        

        # Assuming x and u are your tensors
        #print(x.shape)

        
        
        
        y = x[:, :, :, 1].unsqueeze(-1)
        xsliced = x[:, :, :, 0].unsqueeze(-1)
        xnew = torch.cat((xsliced, y), dim=-1)
        
        #print('xnew',xnew.shape)
        u,v,p,T=self.model(xnew)
        nx=u.size(1)
        ny=u.size(2)
        dx = 10 / (nx)
        dy=1/ny
       # Central difference for interior points for u_x
        u_x_internal = (u[:, 2:, 1:-1,:] - u[:, :-2, 1:-1,:]) / (2 * dx)
        u_y_internal= (u[:, 1:-1,2:,:] - u[:, 1:-1,:-2,:]) / (2 * dy)
        u_xx_internal= (u[:, 2:, 1:-1,:] - 2 * u[:, 1:-1, 1:-1,:] + u[:, :-2, 1:-1,:]) / (dx ** 2)
        u_yy_internal=(u[:, 1:-1,2:,:] - 2 * u[:, 1:-1, 1:-1,:] + u[:, 1:-1,:-2,:]) / (dy ** 2)
        
        v_x_internal = (v[:, 2:, 1:-1,:] - v[:, :-2, 1:-1,:]) / (2 * dx)
        v_y_internal= (v[:, 1:-1,2:,:] - v[:, 1:-1,:-2,:]) / (2 * dy)
        v_xx_internal= (v[:, 2:, 1:-1,:] - 2 * v[:, 1:-1, 1:-1,:] + v[:, :-2, 1:-1,:]) / (dx ** 2)
        v_yy_internal=(v[:, 1:-1,2:,:] - 2 * v[:, 1:-1, 1:-1,:] + v[:, 1:-1,:-2,:]) / (dy ** 2)
        
        T_x_internal = (T[:, 2:, 1:-1,:] - T[:, :-2, 1:-1,:]) / (2 * dx)
        T_y_internal= (T[:, 1:-1,2:,:] - T[:, 1:-1,:-2,:]) / (2 * dy)
        T_xx_internal= (T[:, 2:, 1:-1,:] - 2 * T[:, 1:-1, 1:-1,:] + T[:, :-2, 1:-1,:]) / (dx ** 2)
        T_yy_internal=(T[:, 1:-1,2:,:] - 2 * T[:, 1:-1, 1:-1,:] + T[:, 1:-1,:-2,:]) / (dy ** 2)

        p_x_internal=(p[:, 2:, 1:-1,:] - p[:, :-2, 1:-1,:]) / (2 * dx)
        p_y_internal=(p[:, 1:-1,2:,:] - p[:, 1:-1,:-2,:]) / (2 * dy)
        # Continuity Loss
        u_internal=u[:, 1:-1, 1:-1, :]
        

        v_internal=v[:, 1:-1, 1:-1, :]
        

        
        

        
        udiff = (u[:, 1:-1, 1:-1, :] - u[:, :-2, 1:-1, :]) * dy
        vdiff = (v[:, 1:-1, 1:-1, :] - v[:, 1:-1, :-2, :]) * dx  # Adjusted slice along y-axis  
        
        L_cont1 = (udiff + vdiff)

        f=torch.zeros(u_x_internal.shape,device=u.device)
        L_cont=F.mse_loss(L_cont1,f)
        # Momentum Loss (u component and v component)
        #L_mom_u1=((u*u)[:, 1:, 1:-1,:]-(u*u)[:, :-1, 1:-1,:])*dy + ((u*v)[:, 1:-1,1:,:]-(u*v)[:, 1:-1,:-1,:])*dx +(p[:, 1:, 1:-1,:]-p[:, :-1, 1:-1,:])*(dy*self.beta)-self.nu*(u_xx_internal + u_yy_internal)*dx*dy
        L_mom_u1 = (u_internal*u_x_internal + v_internal*u_y_internal + self.beta*p_x_internal - self.nu*(u_xx_internal + u_yy_internal))
        L_mom_u=F.mse_loss(L_mom_u1,f)
        L_mom_v1 = (u_internal*v_x_internal + v_internal*v_y_internal + self.beta*p_y_internal - self.nu*(v_xx_internal + v_yy_internal))
        L_mom_v=F.mse_loss(L_mom_v1,f)
        #Heat Advection Equation Loss
        L_heat1= (u_internal*T_x_internal + v_internal*T_y_internal - self.alpha*(T_xx_internal + T_yy_internal))
        L_heat=F.mse_loss(L_heat1,f)
        u_avg=10**-3
        u_inlet_expected = u_avg*1.5*(1 - 4*((y[:, 0,:, :] ).pow(2)))
        T_wall=350
        T_center=293
        T_inlet_expected = T_wall + (T_center - T_wall) * (1 - 4*((y[:, 0,:, :] ).pow(2)))

        #Loss due to boundary conditions
        #Loss due to boundary conditions
        left_T_bc1 = T[:, 0, :, :]
        left_T_bc=F.mse_loss(left_T_bc1,T_inlet_expected)
        #right_T_bc = T[:, :, -1, :].pow(2).sum()
        top_T_bc1 = T[:, :, 0, :]
        T_wallvec=T_wall*torch.ones_like(top_T_bc1,device=u.device)
        top_T_bc=F.mse_loss(top_T_bc1,T_wallvec)
        bottom_T_bc1 = T[:,:, -1, :]
        bottom_T_bc=F.mse_loss(bottom_T_bc1,T_wallvec)
        #print(T_wallvec)

        boundaryzero=torch.zeros(u[:, :, 0, :].shape,device=u.device)

        left_u_bc = F.mse_loss(u[:, 0, :, :],u_inlet_expected)
        #print(u_inlet_expected)
        #right_u_bc = u[:, :, -1, :].pow(2).sum()
        top_u_bc = F.mse_loss(u[:, :, 0, :],boundaryzero)
        bottom_u_bc = F.mse_loss(u[:, :, -1, :],boundaryzero)

        vzeroleft=torch.zeros(v[:, 0,:, :].shape,device=u.device)
        left_v_bc = F.mse_loss(v[:, 0,:, :],vzeroleft)
        #right_v_bc = v[:, :, -1, :].pow(2).sum()
        top_v_bc = F.mse_loss(v[:, :,0, :],boundaryzero)
        bottom_v_bc = F.mse_loss(v[:, :,-1, :],boundaryzero)

        #left_p_bc = p[:, :, 0, :].pow(2).sum()
        right_p_bc = F.mse_loss(p[:, -1,:, :],vzeroleft)
        #top_p_bc = p[:, 0, :, :].pow(2).sum()
        #bottom_p_bc = p[:, -1, :, :].pow(2).sum()
        bc_loss=torch.norm(left_T_bc+top_T_bc+bottom_T_bc+2*left_u_bc+top_u_bc+bottom_u_bc+2*left_v_bc+top_v_bc+bottom_v_bc+right_p_bc)
        ###integral control volume soft constraints
        #first temperature: 
        #inletTemp=(u[:,0,:,:]*T[:,0,:,:]).sum()
        #outletTemp=(u[:,-1,:,:]*T[:,-1,:,:]).sum()
        #TempAdvection=inletTemp-outletTemp
        #TempAdvectionLoss=F.mse_loss(TempAdvection,torch.zeros_like(TempAdvection))
        #velocity only in x direction
        #inletU=(u[:,0,:,:]*u[:,0,:,:]).sum()
        #outletU=(u[:,-1,:,:]*u[:,-1,:,:]).sum()
       #expected=(u_inlet_expected*u_inlet_expected).sum()
        
        
        #flowLoss1=F.mse_loss(inletU,expected)
        #flowLoss2=F.mse_loss(outletU,expected)
        #integral_loss=torch.norm(flowLoss2)
        # Total Loss
        L = torch.norm(L_mom_u + L_mom_v+L_cont+L_heat+2*bc_loss)
        
        #print(L)
        return L
# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = True
        self.h5 = False
        self._load_file()

    def _load_file(self):

        if self.file_path[-3:] == '.h5':
            self.data = h5py.File(self.file_path, 'r')
            self.h5 = True

        else:
            try:
                self.data = scipy.io.loadmat(self.file_path)
            except:
                self.data = h5py.File(self.file_path, 'r')
                self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if self.h5:
            x = x[()]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        # Dimension and Lp-norm type are positive
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]
        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = (h**(self.d/self.p))*torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)  
class VeloLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(VeloLoss, self).__init__()
        # Dimension and Lp-norm type are positive
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y, mask):
        num_examples = x.size()[0]
        
        # Compute the difference only for the fluid domain points (where mask == 1)
        diff = torch.where(mask.unsqueeze(-1) == 1, x - y, torch.zeros_like(x))
        
        # Compute the Lp-norm of the difference
        diff_norms = torch.norm(diff.reshape(num_examples, -1), self.p, 1)
        
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms)
            else:
                return torch.sum(diff_norms)
        
        return diff_norms

    def __call__(self, x, y, mask):
        return self.rel(x, y, mask)

# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

class MMDStatistic:
    r"""The *unbiased* MMD test of :cite:`gretton2012kernel`.
    The kernel used is equal to:
    .. math ::
        k(x, x') = \sum_{j=1}^k e^{-\alpha_j\|x - x'\|^2},
    for the :math:`\alpha_j` proved in :py:meth:`~.MMDStatistic.__call__`.
    Arguments
    ---------
    n_1: int
        The number of points in the first sample.
    n_2: int
        The number of points in the second sample."""

    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

        # The three constants used in the test.
        self.a00 = 1. / (n_1 * (n_1 - 1))
        self.a11 = 1. / (n_2 * (n_2 - 1))
        self.a01 = - 1. / (n_1 * n_2)

    def __call__(self, sample_1, sample_2, alphas, ret_matrix=False):
        r"""Evaluate the statistic.
        The kernel used is
        .. math::
            k(x, x') = \sum_{j=1}^k e^{-\alpha_j \|x - x'\|^2},
        for the provided ``alphas``.
        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, of size ``(n_1, d)``.
        sample_2: variable of shape (n_2, d)
            The second sample, of size ``(n_2, d)``.
        alphas : list of :class:`float`
            The kernel parameters.
        ret_matrix: bool
            If set, the call with also return a second variable.
            This variable can be then used to compute a p-value using
            :py:meth:`~.MMDStatistic.pval`.
        Returns
        -------
        :class:`float`
            The test statistic.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""
        sample_12 = torch.cat((sample_1, sample_2), 0)
        distances = pdist(sample_12, sample_12, norm=2)

        kernels = None
        for alpha in alphas:
            kernels_a = torch.exp(- alpha * distances ** 2)
            if kernels is None:
                kernels = kernels_a
            else:
                kernels = kernels + kernels_a

        k_1 = kernels[:self.n_1, :self.n_1]
        k_2 = kernels[self.n_1:, self.n_1:]
        k_12 = kernels[:self.n_1, self.n_1:]

        mmd = (2 * self.a01 * k_12.sum() +
               self.a00 * (k_1.sum() - torch.trace(k_1)) +
               self.a11 * (k_2.sum() - torch.trace(k_2)))
        if ret_matrix:
            return mmd, kernels
        else:
            return mmd

# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c
