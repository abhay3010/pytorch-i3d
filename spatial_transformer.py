import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
from resizer import ConvUnit

"""Spatial transformer module class to add to the resizer network"""

class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, in_res = 112):
        super(SpatialTransformer, self).__init__()
        self.localization = nn.Sequential(
        nn.Conv2d(self.in_channels, 16, kernel_size=[7,7], stride=[1,1],padding=3),
        nn.MaxPool2d(3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 8, kernel_size = 5, padding=2),
        nn.MaxPool2d(2, stride=2, padding=[0,0,0]),
        nn.ReLU()
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(int(((in_res/4)**2)), 32), 
            nn.ReLU(),
            nn.Linear(32, 3*2)
        )

class SpatialTransformer3D(nn.Module):
    def __init__(self, in_channels, in_res=112, in_time=32):
        self.in_channels = in_channels
        self.in_time = in_time
        self.in_res = in_res
        super(SpatialTransformer3D, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv3d(self.in_channels, 16, kernel_size=[7,7,7], stride=[1,1,1],padding=[3, 3,3]),
            nn.MaxPool3d([2,3,3], stride=[2,2,2], padding=[0,1,1]),
            nn.ReLU(),
            nn.Conv3d(16, 8, kernel_size=[5,5,5], padding=[2,2,2]),
            nn.MaxPool3d([2,2,2], stride=[2,2,2], padding=[0,0,0]),
            nn.ReLU()
        

        )
        #at this point the spatial dimensions are input/4 and the temporal dimensions are the same
        self.fc_loc = nn.Sequential(
            nn.Linear(int((in_time/4)*8*((in_res/4)**2)), 32), 
            nn.ReLU(),
            nn.Linear(32, 3*4)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))
    def forward(self, x):
        xs =  self.localization(x)
        xs = xs.view([-1,int((self.in_time/4)*8*((self.in_res/4)**2)) ])
        theta = self.fc_loc(xs)
        theta = theta.view(-1,3,4)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x



if __name__ == '__main__':
    c = SpatialTransformer3D(3, in_res=112, in_time=32)
    summary(c, (3,32, 70, 70), batch_size=1)

