import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

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
from resizer import ResizerMainNetwork

from virat_dataset import Virat as Dataset

def eval(resizer_model, model_path, root, classes_file):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    val_dataset = Dataset(root, "test",classes_file,resize_shape=(60,60), transforms=None)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2) 
    resizer = ResizerMainNetwork(3, 32, (112, 112))
    resizer.load_state_dict(torch.load(resizer_model))
    resizer.to(device)
   
    i3d = InceptionI3d(26,mode="32x112", in_channels=3)
    state_dict = torch.load(model_path)
    new_dict = dict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_dict[k[7:]] = v
        else:
            print(k)
            new_dict[k] = v
    i3d.load_state_dict(new_dict)
    i3d.to(device)
    if torch.cuda.device_count()>1:
        i3d = nn.DataParallel(i3d)
        resizer = nn.DataParallel(resizer)
    i3d.train(False)
    resizer.train(False)
    predictions = list()
    trues = list()

    count = 0
    print("Beginning evaluation for resizer model ", resizer_model, "code model ", model_path)
    for batch, labels in val_dataloader:
        count+=1
        inputs = Variable(batch.to(device))
        out = resizer(inputs)
        v = torch.sigmoid(i3d(out))
        for y_t, y_p in zip(labels, v):
            p = np.array([1 if z >=0.5 else 0 for z in y_p])
            predictions.append(p)
            trues.append(y_t.numpy())    

    #print(trues, predictions)
    f1_macro = f1_score(trues, predictions, average='macro')
    f1_micro = f1_score(trues, predictions, average='micro')
    accuracy = accuracy_score(trues, predictions)
    

    print(f1_macro, f1_micro, accuracy)
    return f1_macro, f1_micro, accuracy

def main():
    i3d_model = "/virat-vr/models/pytorch-i3d/v7_bilinear_32_112004400.pt"
    model_list = ['bilinear_32_resizer_v2_60x60000010.pt', 'bilinear_32_resizer_v2_60x60000015.pt']
    
    for model in model_list:
       f1_macro, f1_micro, accuracy = eval('/virat-vr/models/pytorch-i3d/'+ model, i3d_model, "/mnt/data/TinyVIRAT/", "classes.txt")
       print ("{0} , f1_macro : {1}, f1_micro {2}, Accuracy {3}".format(model,f1_macro, f1_micro, accuracy))
if __name__ == '__main__':
    main()  