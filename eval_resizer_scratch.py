import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse
from torchvision.utils import save_image

# parser = argparse.ArgumentParser()
# parser.add_argument('-mode', type=str, help='rgb or flow')
# parser.add_argument('-save_model', type=str)
# parser.add_argument('-root', type=str)

# args = parser.parse_args()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np


import numpy as np

from i3d import InceptionI3d
from resizer import ResizerMainNetworkV3

from virat_dataset import Virat as Dataset
from torchsummary import summary
from virat_dataset import collate_tensors

def eval(model_list,time_d,i3d_mode, root, classes_file):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    val_dataset = Dataset(root, "test",classes_file, resize=False, transforms=None, load_all=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_tensors) 
    
    for model_path in model_list:
        model = load_models(model_path, i3d_mode, time_d)
        predictions = list()
        trues = list()
        print("Beginning evaluation for resizer model ", model_path)
        for batch, labels in val_dataloader:
            inputs = Variable(batch.to(device))
            out = model(inputs)
            v = torch.sigmoid(out)
            for y_t, y_p in zip(labels, v):
                p = np.array([1 if z >=0.5 else 0 for z in y_p])
                predictions.append(p)
                trues.append(y_t.numpy())    
        f1_macro = f1_score(trues, predictions, average='macro')
        f1_micro = f1_score(trues, predictions, average='micro')
        accuracy = accuracy_score(trues, predictions)
    

        print("{0} , f1_macro : {1}, f1_micro {2}, Accuracy {3}".format(model_path,f1_macro, f1_micro, accuracy))
    

def main():
    mpath = "/virat-vr/models/pytorch-i3d/"
    model_list = [mpath+"bilinear_32_resizer_trained_together000004_prev.pt", mpath+ "bilinear_32_resizer_trained_together000005_prev.pt"]
    root = "/mnt/data/TinyVIRAT"
    time_d = 32
    i3d_mode = "32x112"
    eval(model_list,time_d,i3d_mode, root, "classes.txt")

def load_models(model_path, i3d_mode, time_d):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    i3d = InceptionI3d(26,mode=i3d_mode, in_channels=3)
    resizer = ResizerMainNetworkV3(3, int(i3d_mode.split('x')[0]), time_d)
    final_model = nn.Sequential(OrderedDict([
        ('resizer',resizer),
        ('i3d',i3d)
    ]))
    final_model.load_state_dict(torch.load(model_path))




if __name__ == '__main__':
    main()
