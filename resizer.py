import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np

"""
Resizer module class to add to the video network
"""

#No learnable parameters in the ResizerBlock

#Separate class from Unit3D in order to apply Batchnorm before relu
class ConvUnit(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 lru=True,
                 norm='BatchNorm3d',
                 use_bias=False):
        
        """Initializes Unit3D module."""
        super(ConvUnit, self).__init__()
        
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.use_bias = use_bias
        #self.name = name
        #self.padding = padding
        
        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self.output_channels,
                                kernel_size=self.kernel_shape,
                                stride=self.stride,
                                padding=0, # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self.use_bias)
        
        
        self.bn = nn.__dict__[norm](self.output_channels, eps=0.001, momentum=0.01) if norm is not None else None
        self.act = nn.LeakyReLU(0.2) if lru else None
        

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_shape[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_shape[dim] - (s % self.stride[dim]), 0)

            
    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        #print x.size()        

        x = self.conv3d(x)
        if self.act is not None:
            x = self.act(x)
        if self.bn is not None:
            x = self.bn(x)
        
        return x

class ResizerBlock(nn.Module):
    def __init__(self, output_shape, normalize):
        if len(output_shape) != 3:
            raise ValueError("Expects output dimension of shape 3*3*3 of the format TxWxL")
        self.output_shape = output_shape
        self.normalize = normalize
        
        super(ResizerBlock, self).__init__()
    def forward(self, x):
        """Given input x , resize the input using the given mode to the output shape:
        """
        y = F.interpolate(x, size=self.output_shape,mode='trilinear', align_corners=True) #Setting align corners as true as we want the corners of our image aligned. 
        if self.normalize:
            y = y*2. - 1
        return y
class ResizerMainNetwork(nn.Module):
    def __init__(self, in_channels,n_frames, scale_shape,num_resblocks=1, skip=False):
        self.in_channels = in_channels
        self.r = num_resblocks
        self.scale_shape = scale_shape
        self.nframes = n_frames
        self.skip = skip
        super(ResizerMainNetwork, self).__init__()
        self.skip_resizer =  ResizerBlock((self.nframes,)+self.scale_shape, False)
        if not self.skip:
            self.c1 = ConvUnit(in_channels=self.in_channels, output_channels=self.nframes, kernel_shape=[7, 7, 7],  norm=None)
            #revisit size of this unit as it is inconsitent between paper and diagram
            self.c2 = ConvUnit(in_channels=self.nframes, kernel_shape = [1,1,1], output_channels=self.nframes)
            self.resizer_first = ResizerBlock((self.nframes,) + self.scale_shape, False)
            self.residual_blocks = make_residuals(num_resblocks, self.nframes)
            self.c3 = ConvUnit(in_channels=self.nframes, kernel_shape=[3,3,3], output_channels=self.nframes, lru=False)
            self.c4 = ConvUnit(in_channels=self.nframes, kernel_shape=[1,3,3], output_channels=self.in_channels, lru=False, norm=None)
        
    def forward(self, x):
        # print("input shape", x.shape)
        residual = self.skip_resizer(x)
        if self.skip:
            return residual
        else:

        # print("resizer_shape", out.shape)
            out = self.c1(x)
            # print("conv shape", out.shape)

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
            # print(out.shape)
            out+=residual
            return out

class ResizerMainNetworkV2(nn.Module):
    def __init__(self, in_channels,n_frames, scale_shape,num_resblocks=1, skip=False):
        self.in_channels = in_channels
        self.r = num_resblocks
        self.scale_shape = scale_shape
        self.nframes = n_frames
        self.skip = skip
        super(ResizerMainNetworkV2, self).__init__()
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
        # print("input shape", x.shape)
        residual = self.skip_resizer(x)
        if self.skip:
            return residual
        else:

        # print("resizer_shape", out.shape)
            out = self.c1(x)
            # print("conv shape", out.shape)

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
            # print(out.shape)
            out+=residual
            return out

class ResizerMainNetworkV3(nn.Module):
    def __init__(self, in_channels,n_frames, scale_shape,num_resblocks=1, skip=False):
        self.in_channels = in_channels
        self.r = num_resblocks
        self.scale_shape = scale_shape
        self.nframes = n_frames
        self.skip = skip
        super(ResizerMainNetworkV3, self).__init__()
        self.skip_resizer =  ResizerBlock((self.nframes,)+self.scale_shape, False)
        if not self.skip:
            self.c1 = ConvUnit(in_channels=self.in_channels, output_channels=16, kernel_shape=[7, 7, 7],  norm=None)
            #revisit size of this unit as it is inconsitent between paper and diagram
            self.c2 = ConvUnit(in_channels=16, kernel_shape = [1,1,1], output_channels=16)
            self.resizer_first = ResizerBlock((self.nframes,) + self.scale_shape, False)
            self.c3 = ConvUnit(in_channels=16, kernel_shape=[3,3,3], output_channels=16, lru=False)
            self.c4 = ConvUnit(in_channels=16, kernel_shape=[7,7,7], output_channels=self.in_channels, lru=False, norm=None)
        
    def forward(self, x):
        # print("input shape", x.shape)
        residual = self.skip_resizer(x)
        if self.skip:
            return residual
        else:

        # print("resizer_shape", out.shape)
            out = self.c1(x)
            # print("conv shape", out.shape)

            out = self.c2(out)
            # print("conv2 shape", out.shape)
            out =  self.resizer_first(out)
            # print("in resizer shape", out.shape)
            # residual_skip = out
            # out = self.residual_blocks(out)
            out = self.c3(out)
            # out+=residual_skip
            # print(out.shape)        
            out = self.c4(out)
            # print(out.shape)
            out+=residual
            return out
class ResizerWithTimeCompression(nn.Module):
    def __init__(self, in_channels,n_frames,n_output_frames, scale_shape,num_resblocks=1, skip=False):
        self.in_channels = in_channels
        self.r = num_resblocks
        self.scale_shape = scale_shape
        self.n_input_frames = n_frames
        self.n_output_frames = n_output_frames
        self.skip = skip
        super(ResizerWithTimeCompression, self).__init__()
        self.skip_resizer =  ResizerBlock((self.n_output_frames,)+self.scale_shape, False)
        if not self.skip:
            self.c1 = ConvUnit(in_channels=self.in_channels, output_channels=16, kernel_shape=[7, 7, 7],  norm=None)
            #revisit size of this unit as it is inconsitent between paper and diagram
            self.c2 = ConvUnit(in_channels=16, kernel_shape = [1,1,1], output_channels=16)
            self.resizer_first = ResizerBlock((self.n_output_frames,) + self.scale_shape, False)
            self.c3 = ConvUnit(in_channels=16, kernel_shape=[3,3,3], output_channels=16, lru=False)
            self.c4 = ConvUnit(in_channels=16, kernel_shape=[7,7,7], output_channels=self.in_channels, lru=False, norm=None)
        
    def forward(self, x):
        # print("input shape", x.shape)
        residual = self.skip_resizer(x)
        if self.skip:
            return residual
        else:

        # print("resizer_shape", out.shape)
            out = self.c1(x)
            # print("conv shape", out.shape)

            out = self.c2(out)
            # print("conv2 shape", out.shape)
            out =  self.resizer_first(out)
            # print("in resizer shape", out.shape)
            # residual_skip = out
            # out = self.residual_blocks(out)
            out = self.c3(out)
            # out+=residual_skip
            # print(out.shape)        
            out = self.c4(out)
            # print(out.shape)
            out+=residual
            return out

class ResidualBlock(nn.Module):
    def __init__(self,num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(num_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(num_channels)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out+=residual
        return out
def make_residuals(r, in_channels):
    residuals = []
    for i in range(r):
        b = ResidualBlock(in_channels)
        residuals.append(b)
    return nn.Sequential(*residuals)

def main():
    resizer_network = ResizerWithTimeCompression(3,32, (112,112) )
    summary(resizer_network, (3, 64, 28, 28), batch_size=2)
if __name__ == '__main__':
    main()
    