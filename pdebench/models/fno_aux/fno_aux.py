from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
import pdb



class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            # Number of Fourier modes to multiply, at most floor(N/2) + 1
            modes1
        )
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO2d(nn.Module):
    def __init__(self, num_channels, modes1=12, modes2=12, width=20, initial_step=10):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x, y, c)
        output: the solution of the next timestep
        output shape: (batchsize, x, y, c)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(initial_step * num_channels + 2, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        # Fourier layer
        self.conv0 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2
        )
        self.conv1 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2
        )
        self.conv2 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2
        )
        self.conv3 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2
        )
        # pointwise convolutions
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        # self.fc2 = nn.Linear(128, num_channels)
        self.fc2_primary = nn.Linear(128, num_channels)  # Primary task
        self.fc2_auxiliary = nn.Linear(128, num_channels)  # Auxiliary task

        self.shared_layers = nn.ModuleList([
            self.fc0, self.conv0, self.conv1, self.conv2, self.conv3,
            self.w0, self.w1, self.w2, self.w3, self.fc1
        ])

    def forward(self, x, grid, x_aux, grid_aux):
        # x dim = [b, x1, x2, t, v]
        # x_aux dim = [b * num_aux, x1, x2, t, v]
        # Normalize (time + channels per sample) - treat each auxiliary sample independently
        # XXXX


        with torch.no_grad():
            data_std, data_mean = torch.std_mean(x, dim=(1,2,3), keepdims=True) 
            data_std = data_std + 1e-7 # Orig 1e-7
            data_std_aux, data_mean_aux = torch.std_mean(x_aux, dim=(1,2,3), keepdims=True) 
            data_std_aux = data_std_aux + 1e-7 # Orig 1e-7
        x = (x - data_mean) / (data_std) 
        x_aux = (x_aux - data_mean_aux) / (data_std_aux)

        inp_shape = list(x.shape)
        inp_shape = inp_shape[:-2]
        inp_shape.append(-1)

        inp_shape_aux = list(x_aux.shape)
        inp_shape_aux = inp_shape_aux[:-2]
        inp_shape_aux.append(-1)

        # concatenate the spatial grid to the input
        # x dim = [b, x1, x2, t*v]
        inp = x.reshape(inp_shape)
        inp_aux = x_aux.reshape(inp_shape_aux)
        #inp_aux = x_aux.reshape(B * num_aux, X, Y, T * v)

        x = torch.cat((inp, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2) # [b, x1, x2, t*v] -> [b, t*v, x1, x2]

        x_aux = torch.cat((inp_aux, grid_aux), dim=-1)
        x_aux = self.fc0(x_aux)
        x_aux = x_aux.permute(0, 3, 1, 2) # [b*num_aux, x1, x2, t*v] -> [b*num_aux, t*v, x1, x2]

        # Pad tensor with boundary condition
        x = F.pad(x, [0, self.padding, 0, self.padding])
        x_aux = F.pad(x_aux, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., : -self.padding, : -self.padding]  # Unpad the tensor
        x = x.permute(0, 2, 3, 1) # [b, t*v, x1, x2] -> [b, x1, x2, t*v]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2_primary(x)
        # **Denormalization**
        out_primary = x * data_std.squeeze(-2) + data_mean.squeeze(-2)
        
        # x = self.fc2(x)

        x1_aux = self.conv0(x_aux)
        x2_aux = self.w0(x_aux)
        x_aux = x1_aux + x2_aux
        x_aux = F.gelu(x_aux)

        x1_aux = self.conv1(x_aux)
        x2_aux = self.w1(x_aux)
        x_aux = x1_aux + x2_aux
        x_aux = F.gelu(x_aux)

        x1_aux = self.conv2(x_aux)
        x2_aux = self.w2(x_aux)
        x_aux = x1_aux + x2_aux
        x_aux = F.gelu(x_aux)

        x1_aux = self.conv3(x_aux)
        x2_aux = self.w3(x_aux)
        x_aux = x1_aux + x2_aux

        x_aux = x_aux[..., : -self.padding, : -self.padding]  # Unpad the tensor
        x_aux = x_aux.permute(0, 2, 3, 1) # [b*num_aux, t*v, x1, x2] -> [b*num_aux, x1, x2, t*v]
        x_aux = self.fc1(x_aux)
        x_aux = F.gelu(x_aux)
        x_aux = self.fc2_auxiliary(x_aux) 

        # **Denormalization**
        out_auxiliary = x_aux * data_std_aux.squeeze(-2) + data_mean_aux.squeeze(-2)

        #return x.unsqueeze(-2)
        return out_primary.unsqueeze(-2), out_auxiliary.unsqueeze(-2)  # Maintain expected dimensions


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            # Number of Fourier modes to multiply, at most floor(N/2) + 1
            modes1
        )
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights3 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights4 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        # Return to physical space
        return torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))


class FNO3d(nn.Module):
    def __init__(
        self, num_channels, modes1=8, modes2=8, modes3=8, width=20, initial_step=10
    ):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(initial_step * num_channels + 3, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv1 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv2 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv3 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2_primary = nn.Linear(128, num_channels)  # Primary task
        self.fc2_auxiliary = nn.Linear(128, num_channels)  # Auxiliary task

        self.shared_layers = nn.ModuleList([
            self.fc0, self.conv0, self.conv1, self.conv2, self.conv3,
            self.w0, self.w1, self.w2, self.w3, self.fc1,
            self.bn0, self.bn1, self.bn2, self.bn3
        ])

    def forward(self, x, grid, x_aux, grid_aux):
        # x dim = [b, x1, x2, x3, t, v]
        # x_aux dim = [b * num_aux, x1, x2, t, v]
        # Normalize (time + channels per sample) - treat each auxiliary sample independently
        # XXXX

        with torch.no_grad():
            data_std, data_mean = torch.std_mean(x, dim=(1,2,3,4), keepdims=True) 
            data_std = data_std + 1e-7 # Orig 1e-7
            data_std_aux, data_mean_aux = torch.std_mean(x_aux, dim=(1,2,3,4), keepdims=True) 
            data_std_aux = data_std_aux + 1e-7 # Orig 1e-7
        x = (x - data_mean) / (data_std) 
        x_aux = (x_aux - data_mean_aux) / (data_std_aux)

        inp_shape = list(x.shape)
        inp_shape = inp_shape[:-2]
        inp_shape.append(-1)

        inp_shape_aux = list(x_aux.shape)
        inp_shape_aux = inp_shape_aux[:-2]
        inp_shape_aux.append(-1)

        inp = x.reshape(inp_shape)
        inp_aux = x_aux.reshape(inp_shape_aux)

        # x dim = [b, x1, x2, x3, t*v]
        x = torch.cat((inp, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x_aux = torch.cat((inp_aux, grid_aux), dim=-1)
        x_aux = self.fc0(x_aux)
        x_aux = x_aux.permute(0, 4, 1, 2, 3)

        # pad the domain if input is non-periodic
        x = F.pad(x, [0, self.padding])
        x_aux = F.pad(x_aux, [0, self.padding])
        # x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])
        # x_aux = F.pad(x_aux, [0, self.padding, 0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., : -self.padding]
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2_primary(x)
        # **Denormalization**
        out_primary = x * data_std.squeeze(-2) + data_mean.squeeze(-2)

        x1_aux = self.conv0(x_aux)
        x2_aux = self.w0(x_aux)
        x_aux = x1_aux + x2_aux
        x_aux = F.gelu(x_aux)

        x1_aux = self.conv1(x_aux)
        x2_aux = self.w1(x_aux)
        x_aux = x1_aux + x2_aux
        x_aux = F.gelu(x_aux)

        x1_aux = self.conv2(x_aux)
        x2_aux = self.w2(x_aux)
        x_aux = x1_aux + x2_aux
        x_aux = F.gelu(x_aux)

        x1_aux = self.conv3(x_aux)
        x2_aux = self.w3(x_aux)
        x_aux = x1_aux + x2_aux

        x_aux = x_aux[..., : -self.padding]  # Unpad the tensor
        x_aux = x_aux.permute(0, 2, 3, 4, 1) # [b*num_aux, t*v, x1, x2] -> [b*num_aux, x1, x2, t*v]
        x_aux = self.fc1(x_aux)
        x_aux = F.gelu(x_aux)
        x_aux = self.fc2_auxiliary(x_aux) 
        out_auxiliary = x_aux * data_std_aux.squeeze(-2) + data_mean_aux.squeeze(-2)
        return out_primary.unsqueeze(-2), out_auxiliary.unsqueeze(-2)
