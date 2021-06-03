import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
from resizer import ConvUnit
from virat_dataset import Virat as Dataset
from i3d import InceptionI3d

class TwoStreamNetwork(nn.Module):
    def __init__(self, model_to_train, model_to_help, num_classes):
        self.num_classes = num_classes
        super(TwoStreamNetwork, self).__init__()
        self.model_to_train = model_to_train
        self.model_to_help = model_to_help
        
    def forward(self, x):
        #Given input of shape CxTxHxW change to C*TxHxW and then apply the affine transformation
        x1 = self.model_to_train(x)
        x2 = self.model_to_help(x)
       
        return (x1 + x2)/2.

    def get_parameters_to_train(self):
        return list(self.model_to_train.parameters()) + list(self.final_layer.parameters())


def mode_summary():
    i3d = InceptionI3d(26, in_channels=3)
    model = TwoStreamNetwork(i3d, i3d, 26)
    summary(model, (3,32,112,112), batch_size=12)


if __name__ == '__main__':
    mode_summary()
        
        
