import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import _random
from resizer import ConvUnit
from virat_dataset import Virat as Dataset
from resizer import make_residuals, ConvUnit, ResizerBlock
import random
# architecture to experiment with the calculation and application of theta for the transformation. 
#we need to sample at point 1 and apply at another point. And consider all strategies of doing so.

class TransformerWithResizer(nn.Module):
    def __init__(self, in_channels,n_frames, scale_shape,in_res=112, num_resblocks=1, skip=False):
        super(TransformerWithResizer, self).__init__()
        self.in_channels = in_channels
        self.r = num_resblocks
        self.scale_shape = scale_shape
        self.in_res = in_res
        self.nframes = n_frames
        self.skip = skip
        
        self.localization = nn.Sequential(
        nn.Conv2d(self.in_channels, 16, kernel_size=[5,5], stride=[1,1],padding=2),
        nn.MaxPool2d(3, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.Tanh(),
        nn.Conv2d(16, 8, kernel_size = 3, padding=1),
        nn.MaxPool2d(2, stride=2, padding=[0,0]),
        nn.BatchNorm2d(8),
        nn.Tanh())
        self.fc_loc = nn.Sequential(
            nn.Linear(int(8*((in_res/4)**2)), 32), 
            nn.Tanh(),
            nn.Linear(32, 3*2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
        self.skip_resizer =  ResizerBlock((self.nframes,)+self.scale_shape, False)
        if not self.skip:
            self.c1 = ConvUnit(in_channels=self.in_channels, output_channels=16, kernel_shape=[7, 7, 7],  norm=None)
            #revisit size of this unit as it is inconsitent between paper and diagram
            self.c2 = ConvUnit(in_channels=16, kernel_shape = [1,1,1], output_channels=16)
            self.resizer_first = ResizerBlock((self.nframes,) + self.scale_shape, False)
            self.residual_blocks = make_residuals(num_resblocks, 16)
            self.c3 = ConvUnit(in_channels=16, kernel_shape=[3,3,3], output_channels=16, lru=False)
            self.c4 = ConvUnit(in_channels=16, kernel_shape=[3,3,3], output_channels=self.in_channels, lru=False, norm=None)
    def forward(self, x):
        theta = self.get_theta(x)
        x = self.apply_theta(theta, x)

        #print("input shape", x.shape)
        residual = self.skip_resizer(x)
        #print(residual.shape)
        #theta = self.get_theta(residual)
        
        if self.skip:
            return residual
        else:

        # print("resizer_shape", out.shape)
            out = self.c1(x)
            # print("conv shape", out.s

            out = self.c2(out)
            # print("conv2 shape", out.shape)
            out =  self.resizer_first(out)
            
            # print("in resizer shape", out.shape)
            residual_skip = out
            out = self.residual_blocks(out)
            out = self.c3(out)
            out+=residual_skip
            # print(out.shape)        
            out = self.c4(out)
            #print(out.shape)
            out+=residual
            # print("out", out.shape)
            # theta = self.get_theta(out)
            # out = self.apply_theta(theta, out)
            return out

    def get_theta(self, x):
        #Given input of shape CxTxHxW change to C*TxHxW and then apply the affine transformation
        c = x.shape[1]
        b = x.shape[0]
        t = x.shape[2]
        h = x.shape[3]
        w = x.shape[4]
        # print(b,c,t,h,w)
        x_view = x.view(-1,c,h,w)
        xs =  self.localization(x_view)
        #print(xs.shape)
        xs = xs.view([-1,int(8*((self.in_res/4)**2)) ])
        theta = self.fc_loc(xs)
        theta = theta.view(-1,2,3)
        # if random.uniform(0,1) <=0.03: 
        print("theta", theta.detach().cpu().numpy()[0])
        return theta
    def apply_theta(self, theta, x):
        c = x.shape[1]
        b = x.shape[0]
        t = x.shape[2]
        h = x.shape[3]
        w = x.shape[4]
        #print(theta.shape)
        x_view = x.view(-1,c,h,w)
        grid = F.affine_grid(theta, x_view.size(),align_corners=False)
        x_view = F.grid_sample(x_view, grid, align_corners=False)
        o = x_view.view(b,c,t,h,w)
        return o


class TransformerWithResizer3D(nn.Module):
    def __init__(self, in_channels,n_frames, scale_shape,in_res=112, num_resblocks=1, skip=False):
        super(TransformerWithResizer3D, self).__init__()
        self.in_channels = in_channels
        self.r = num_resblocks
        self.scale_shape = scale_shape
        self.in_res = in_res
        self.nframes = n_frames
        self.skip = skip
        
        self.localization = nn.Sequential(
        nn.Conv3d(self.in_channels, 16, kernel_size=[5,5,5], stride=[1,1],padding=2),
        nn.MaxPool3d(3, stride=2, padding=1),
        nn.BatchNorm3d(16),
        nn.Tanh(),
        nn.Conv2d(16, 8, kernel_size = 3, padding=1),
        nn.MaxPool3d(2, stride=2, padding=[0,0]),
        nn.BatchNorm3d(8),
        nn.Tanh())
        self.fc_loc = nn.Sequential(
            nn.Linear(int(8*((in_res/4)**2)), 32), 
            nn.Tanh(),
            nn.Linear(32, 4*3)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0,  1, 0, 0, 0, 0, 1, 0], dtype=torch.float))
        
        self.skip_resizer =  ResizerBlock((self.nframes,)+self.scale_shape, False)
        if not self.skip:
            self.c1 = ConvUnit(in_channels=self.in_channels, output_channels=16, kernel_shape=[7, 7, 7],  norm=None)
            #revisit size of this unit as it is inconsitent between paper and diagram
            self.c2 = ConvUnit(in_channels=16, kernel_shape = [1,1,1], output_channels=16)
            self.resizer_first = ResizerBlock((self.nframes,) + self.scale_shape, False)
            self.residual_blocks = make_residuals(num_resblocks, 16)
            self.c3 = ConvUnit(in_channels=16, kernel_shape=[3,3,3], output_channels=16, lru=False)
            self.c4 = ConvUnit(in_channels=16, kernel_shape=[3,3,3], output_channels=self.in_channels, lru=False, norm=None)
    def forward(self, x):
        # theta = self.get_theta(x)
        # x = self.apply_theta(theta, x)

        #print("input shape", x.shape)
        residual = self.skip_resizer(x)
        #print(residual.shape)
        #theta = self.get_theta(residual)
        
        if self.skip:
            return residual
        else:

        # print("resizer_shape", out.shape)
            out = self.c1(x)
            # print("conv shape", out.s

            out = self.c2(out)
            # print("conv2 shape", out.shape)
            out =  self.resizer_first(out)
            
            # print("in resizer shape", out.shape)
            residual_skip = out
            out = self.residual_blocks(out)
            out = self.c3(out)
            out+=residual_skip
            # print(out.shape)        
            out = self.c4(out)
            #print(out.shape)
            out+=residual
            # print("out", out.shape)
            theta = self.get_theta(out)
            out = self.apply_theta(theta, out)
            return out

    def get_theta(self, x):
        xs =  self.localization(x)
        theta = self.fc_loc(xs)
        theta = theta.view(-1,3,4)
        if random.uniform(0,1) <=0.03: 
            print("theta", theta.detach().cpu().numpy()[0])
        return theta
    def apply_theta(self, theta, x):
        grid = F.affine_grid(theta, x.size(),align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    

def main():
    resizer_network = TransformerWithResizer3D(3,32,(112,112),in_res=28, num_resblocks=1 )
    summary(resizer_network, (3, 32, 28, 28), batch_size=2)
    

    
if __name__ == '__main__':
    main()
    