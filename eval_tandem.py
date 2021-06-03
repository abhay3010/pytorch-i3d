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
from spatial_transformer import SpatialTransformer
from two_stream_i3d import TwoStreamNetwork
import numpy as np
import numpy as np
from i3d import InceptionI3d
from resizer import *
from virat_dataset import Virat as Dataset
from torchsummary import summary
from virat_dataset import collate_tensors, load_rgb_frames

def eval(model_path, root, classes_file,v_mode='32x112', input_shape=(112,112), debug=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    val_dataset = Dataset(root, "test",classes_file,num_frames=32, resize=True,resize_shape=(28,28), transforms=None, sample=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=5, pin_memory=True) 
    model = load_model_from_file(model_path, v_mode=v_mode, model_input_shape=input_shape, input_res=28)
    predictions = list()
    trues = list()
    p_logits = list()

    count = 0
    print("Beginning evaluation for tandem model", model_path)
    for batch, labels in val_dataloader:
        count+=1
        batch.to(device)
        v = torch.sigmoid(model(batch))
        for y_t, y_p in zip(labels, v):
            p = np.array([1 if z >=0.5 else 0 for z in y_p])
            predictions.append(p)
            trues.append(y_t.numpy()) 
            p_logits.append(y_p.cpu().detach().numpy())
        print(count)


    #print(trues, predictions)

    f1_macro = f1_score(trues, predictions, average='macro')
    f1_micro = f1_score(trues, predictions, average='micro')
    accuracy = accuracy_score(trues, predictions)    

    print(f1_macro, f1_micro, accuracy)


    return f1_macro, f1_micro, accuracy

def main():
    #i3d_model = "/virat-vr/models/pytorch-i3d/v7_bilinear_32_112004400.pt"
    prefix = 'two_stream_28_deactivated'
    #cpu params
    model_path = "./eval_models/"
    data_root = "./TinyVIRAT/"
    #Gpu params
    model_path = "/virat-vr/models/pytorch-i3d/"
    data_root = "/mnt/data/TinyVIRAT/"
    model_list = list()
    for epoch in range(0, 5):
        model_list.append((prefix+str(epoch).zfill(6)+'.pt', prefix+ 'i3d'+str(epoch).zfill(6)+'.pt'))
    for model in model_list:
       f1_macro, f1_micro, accuracy = eval(model_path+ model, data_root, "classes.txt", debug=False)
       print ("{0} , f1_macro : {1}, f1_micro {2}, Accuracy {3}".format(model,f1_macro, f1_micro, accuracy))


def load_model_from_file(model_path, v_mode="32x112", model_input_shape=(112,112), input_res=28):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    i3d = InceptionI3d(26, mode="32x112", in_channels=3)
    resizer = ResizerMainNetworkV4_3D(3, 32, (112,112),num_resblocks=1)
    model1 =  nn.Sequential(resizer, i3d)
    i3d_2 = InceptionI3d(26, mode="32x112", in_channels=3)
    model2 = nn.Sequential(
        SpatialTransformer(3, in_time=int(v_mode.split('x')[0]), in_res=input_res),
        ResizerMainNetworkV4_2D(3, int(v_mode.split('x')[0]), model_input_shape,num_resblocks=1),
    i3d_2)
    model = TwoStreamNetwork(model2, model1, 26)
    model.load_state_dict(torch.load(model_path, device))
    model.to(device)
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)
    model.train(False)
    return model

    







if __name__ == '__main__':
    main()
