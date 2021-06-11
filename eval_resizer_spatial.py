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
from spatial_transformer import SpatialTransformer


import numpy as np

from i3d import InceptionI3d
from resizer import ResizerMainNetworkV4_3D, ResizerMainNetworkV4_2D

from virat_dataset import Virat as Dataset
from torchsummary import summary
from virat_dataset import collate_tensors, load_rgb_frames
from spatial_resizer import *

def eval(resizer_model, model_path, root, classes_file, num_frames=32, resize_shape=28, model_input_shape=112, num_workers=5, batch_size=16, i3d_mode='32x112',num_resblocks=1, debug=False, confusion="confusion.npy", predictions="predictions.npy", actuals="actuals.npy", logits="logits.npy"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    val_dataset = Dataset(root, "test",classes_file,num_frames=num_frames, resize_shape=(resize_shape, resize_shape), transforms=None, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) 
    # resizer = nn.Sequential(
    #     SpatialTransformer(3, in_time=int(v_mode.split('x')[0]), in_res=56),
    #     ResizerMainNetworkV4_3D(3, int(v_mode.split('x')[0]), (112,112),num_resblocks=1)
        
    # )
    resizer = TransformerWithResizer(3, num_frames, (model_input_shape,model_input_shape), in_res=model_input_shape, num_resblocks=num_resblocks)
    # resizer = SpatialTransformer(3, in_time=int(v_mode.split('x')[0]), in_res=112)
    resizer.load_state_dict(torch.load(resizer_model))
    resizer.to(device)
   
    i3d = InceptionI3d(26,mode=i3d_mode, in_channels=3)
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
    p_logits = list()

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
            p_logits.append(y_p.cpu().detach().numpy())


    #print(trues, predictions)
    f1_macro = f1_score(trues, predictions, average='macro')
    f1_micro = f1_score(trues, predictions, average='micro')
    accuracy = accuracy_score(trues, predictions)
    f1_samples = f1_score(trues, predictions, average = "samples")    

    print(f1_macro, f1_micro, accuracy, f1_samples)
    pred_np = np.asarray(predictions)
    act_np = np.asarray(trues)
    if debug:
        cf = multilabel_confusion_matrix(trues, predictions)
        np.save(predictions, pred_np)
        np.save(actuals, act_np)
        np.save(logits, np.asarray(p_logits))
        np.save(confusion, cf)

    return f1_macro, f1_micro, accuracy, f1_samples

def eval_model_list(model_prefix, epoch_list, model_path, data_root, classes_file, num_frames=32, resize_shape=28, model_input_shape=112, num_workers=5, batch_size=16, i3d_mode='32x112',num_resblocks=1, debug=False, confusion="confusion.npy", predictions="predictions.npy", actuals="actuals.npy", logits="logits.npy"):
    model_list = list()
    for epoch in epoch_list:
        model_list.append((model_prefix+str(epoch).zfill(6)+'.pt', model_prefix+ 'i3d'+str(epoch).zfill(6)+'.pt'))
    for model, i3d_model in model_list:
        f1_macro, f1_micro, accuracy, f1_samples = eval(model_path + model, model_path+ i3d_model, data_root, classes_file, 
        num_frames=num_frames, resize_shape=resize_shape, model_input_shape=model_input_shape, num_workers=num_workers, batch_size=batch_size,
        i3d_mode=i3d_mode, num_resblocks=num_resblocks, debug=debug, confusion=confusion, predictions=predictions, actuals=actuals, logits=logits)
        print ("{0} , f1_macro : {1}, f1_micro {2}, f1_samples {4},  Accuracy {3}".format(model,f1_macro, f1_micro, accuracy, f1_samples))


def main():
    #i3d_model = "/virat-vr/models/pytorch-i3d/v7_bilinear_32_112004400.pt"
    prefix = 'combined_resizer_56_wider_all_last_in_32'
    model_path = '/virat-vr/models/pytorch-i3d/'
    data_root = '/mnt/data/TinyVIRAT/'
    classes = "classes.txt"
    model_list = list()
    for epoch in range(1,15):
        model_list.append((prefix+str(epoch).zfill(6)+'.pt', prefix+ 'i3d'+str(epoch).zfill(6)+'.pt'))
    for model, i3d_model in model_list:
       f1_macro, f1_micro, accuracy, f1_samples = eval(model_path + model, model_path + i3d_model, data_root,classes, debug=False)
       print ("{0} , f1_macro : {1}, f1_micro {2}, f1_samples {4}, Accuracy {3}".format(model,f1_macro, f1_micro, accuracy, f1_samples))



if __name__ == '__main__':
    main()
