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
from resizer import ResizerWithTimeCompression

from virat_dataset import Virat as Dataset
from torchsummary import summary
from virat_dataset import collate_with_time, load_rgb_frames

def eval(model_list,time_d, root, classes_file):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    val_dataset = Dataset(root, "test",classes_file, resize=False, transforms=None)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_with_time) 
    
    for resizer_path, i3d_path in model_list:
        resizer, i3d = load_models(resizer_path, i3d_path, time_d)
        predictions = list()
        trues = list()
        print("Beginning evaluation for resizer model ", resizer_path, "code model ", i3d_path)
        for batch, labels in val_dataloader:
            inputs = Variable(batch.to(device))
            out = resizer(inputs)
            v = torch.sigmoid(i3d(out))
            for y_t, y_p in zip(labels, v):
                p = np.array([1 if z >=0.5 else 0 for z in y_p])
                predictions.append(p)
                trues.append(y_t.numpy())    
        f1_macro = f1_score(trues, predictions, average='macro')
        f1_micro = f1_score(trues, predictions, average='micro')
        accuracy = accuracy_score(trues, predictions)
    

        print("{0} , f1_macro : {1}, f1_micro {2}, Accuracy {3}".format(model,f1_macro, f1_micro, accuracy))
    

def main():
    #i3d_model = "/virat-vr/models/pytorch-i3d/v7_bilinear_32_112004400.pt"
    model_list = []
    save_model_path = '/virat-vr/models/pytorch-i3d/bilinear_16_resizer_timecompression_v1'
    for i in range(1, 8):
        m = save_model_path + str(i).zfill(6)+'.pt'
        i3d = save_model_path + 'i3d' + str(i).zfill(6)+'.pt'
        model_list.append((m,i3d))

    root = "/mnt/data/TinyVIRAT"
    time_d = 16
    eval(model_list,time_d, root, "classes.txt")
def load_models(resizer_path, i3d_path, time_d):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resizer = ResizerWithTimeCompression(3, time_d, time_d, (112, 112), skip=False)
    resizer.load_state_dict(torch.load(resizer_path))
    resizer.to(device)
   
    i3d = InceptionI3d(26,mode="32x112", in_channels=3)
    state_dict = torch.load(i3d_path)
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
    return resizer, i3d




if __name__ == '__main__':
    main()
