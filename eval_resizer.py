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
from train_resizer import get_resizer_model
from i3d import InceptionI3d
from resizer import *

from virat_dataset import Virat as Dataset
from torchsummary import summary
from virat_dataset import collate_tensors, load_rgb_frames
from spatial_transformer import SpatialTransformer
from spatial_resizer import *

def eval(resizer_model, model_path, root, classes_file, model_type='2d', num_frames=32, resize_shape=28, model_input_shape=112, num_workers=5, batch_size=16, i3d_mode='32x112',num_resblocks=1, debug=False, confusion_file="confusion.npy", predictions_file="predictions.npy", actuals_file="actuals.npy", logits_file="logits.npy"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    val_dataset = Dataset(root, "test",classes_file,num_frames=num_frames, resize=True,resize_shape=(resize_shape,resize_shape), transforms=None)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) 
    resizer = get_resizer_model(model_type, i3d_mode, model_input_shape, num_resblocks=num_resblocks)
    resizer.load_state_dict(torch.load(resizer_model, map_location=device))
    resizer.to(device)
   
    i3d = InceptionI3d(26,mode="32x112", in_channels=3)
    state_dict = torch.load(model_path, map_location=device)
    new_dict = dict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_dict[k[7:]] = v
        else:
            #print(k)
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
    f1_sample = f1_score(trues, predictions, average='samples')
    accuracy = accuracy_score(trues, predictions)
    cf = multilabel_confusion_matrix(trues, predictions)    

    print(f1_macro, f1_micro, accuracy, f1_sample)
    if debug:
        pred_np = np.asarray(predictions)
        act_np = np.asarray(trues)
        cf = multilabel_confusion_matrix(trues, predictions)
        np.save(predictions_file, pred_np)
        np.save(actuals_file, act_np)
        np.save(logits_file, np.asarray(p_logits))
        np.save(confusion_file, cf)


    return f1_macro, f1_micro, accuracy, f1_sample, cf



def main():
    #i3d_model = "/virat-vr/models/pytorch-i3d/v7_bilinear_32_112004400.pt"
    prefix = 'resizerv43d_28_32_3res_'
    model_list = list()
    for epoch in range(3, 25):
        model_list.append((prefix+str(epoch).zfill(6)+'.pt', prefix+ 'i3d'+str(epoch).zfill(6)+'.pt'))
    for model, i3d_model in model_list:
       f1_macro, f1_micro, accuracy, f1_sample = eval('/virat-vr/models/pytorch-i3d/'+ model, '/virat-vr/models/pytorch-i3d/'+ i3d_model, "/mnt/data/TinyVIRAT/", "classes.txt", debug=False)
       print ("{0} , f1_macro : {1}, f1_micro {2}, f1_sample {4}, Accuracy {3}".format(model,f1_macro, f1_micro, accuracy, f1_sample))

def eval_model_list(model_prefix, epoch_list, model_path, data_root, classes_file, model_type='2d', num_frames=32, resize_shape=28, model_input_shape=112, num_workers=5, batch_size=16, i3d_mode='32x112',num_resblocks=1, debug=False, confusion="confusion.npy", predictions="predictions.npy", actuals="actuals.npy", logits="logits.npy"):
    model_list = list()
    for epoch in epoch_list:
        model_list.append((model_prefix+str(epoch).zfill(6)+'.pt', model_prefix+ 'i3d'+str(epoch).zfill(6)+'.pt'))
    max_f1_macro = 0
    confusion_matrix = None
    for model, i3d_model in model_list:
        f1_macro, f1_micro, accuracy, f1_samples, cf = eval(model_path + model, model_path+ i3d_model, data_root, classes_file,model_type=model_type, 
        num_frames=num_frames, resize_shape=resize_shape, model_input_shape=model_input_shape, num_workers=num_workers, batch_size=batch_size,
        i3d_mode=i3d_mode, num_resblocks=num_resblocks, debug=debug, confusion_file=confusion, predictions_file=predictions, actuals_file=actuals, logits_file=logits)
        print ("{0} , f1_macro : {1}, f1_micro {2}, f1_samples {4},  Accuracy {3}".format(model,f1_macro, f1_micro, accuracy, f1_samples))
        if max_f1_macro < f1_macro:
            max_f1_macro = f1_macro
            confusion_matrix = cf
        np.save(confusion, confusion_matrix)

    
    

def test_resizer():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resizer = ResizerMainNetworkV2(3, 32, (112, 112), skip=True)
    #load unnormalized image and then resized image and resizer network should give the same output
    frame_path = 'TinyVIRAT/videos/train/VIRAT_S_000203_07_001341_001458/0.mp4'
    total_frames = 77
    shape = (70,70)
    unresized_frames = load_rgb_frames(frame_path, 0, 32, total_frames, resize=False, normalize=False).transpose([3,0,1,2])
    print(unresized_frames.shape)
    resized_frames = load_rgb_frames(frame_path, 0, 32, total_frames, resize=True, resize_shape=(112, 112), normalize=False).transpose([3, 0, 1, 2])
    print(resized_frames.shape)
    from_network = resizer(torch.from_numpy(unresized_frames).unsqueeze(0)).squeeze(0).numpy()
    print(from_network.shape)
    print(np.min(np.abs(from_network - resized_frames)))

def sample_resizer_output():
    root = './TinyVIRAT/'
    classes_file = "classes.txt"
    resizer_model = 'eval_models/resizer_spatial_corrected_1_56_after_last_last_000016.pt'
    val_dataset = Dataset(root,"test", classes_file, resize=True, resize_shape=(56,56), transforms=None)
    _,val = val_dataset.get_train_validation_split(0.007)
    print(len(val))
    val_dataset_sampled = torch.utils.data.Subset(val_dataset, val)

    x = val_dataset[3]
    print(x[0].shape)
    new_val_dataset_ = Dataset(root,"test", classes_file, resize=True, resize_shape=(56,56), transforms=None, sample=False)
    
    val_dataloader = torch.utils.data.DataLoader(val_dataset_sampled, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    resizer =TransformerWithResizer(3, 32, (112, 112),in_res=112, skip=False, num_resblocks=1)
    # resizer = nn.Sequential(
    #     SpatialTransformer(3, in_time=32, in_res=56),
    #     ResizerMainNetworkV4_2D(3, 32, (112,112),num_resblocks=1)
        
    # )
    resizer_skip = ResizerMainNetworkV2(3,32,(112,112), skip=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resizer.load_state_dict(torch.load(resizer_model, map_location=device))
    
    resizer.to(device)
    resizer_skip.to(device)
    index = 0
    reverse_map = {v:k for k,v in val_dataset.labels_map.items()}
    for batch, label in val_dataloader:

        resized_image_sp = resizer(batch).squeeze(0)
        # print("resizer shape", resized_image_sp.shape)
        resized_normal = new_val_dataset_[val[index]][0]
        # print("resized normal shape", resized_normal.shape)
        permuted_view = (resized_image_sp.permute(1,0,2,3) + 1)/2
        permuted_view_n = (resized_normal.permute(1,0,2,3) +1)/2
        # print("permuted view shape", permuted_view.shape)
        # print("permuted view n shape", permuted_view_n.shape)
        fname = get_fname(label, reverse_map)
        # print(permuted_view.size(0))
        for i in range(permuted_view.size(0)):
            save_image(permuted_view[i], "resized_frames_new/{2}_test_{0}_frame{1}_spatial_56_1res_lastlast_corrected.png".format(index, i, fname))
            #save_image(permuted_view_n[i], "resized_frames_new/{2}_test_{0}_frame{1}_normal.png".format(index, i, fname))
        index+=1
def get_fname(labels, reverse_map):
    labels_np = labels.numpy()
    args = np.where(labels_np == 1)
    print(labels)
    print(args)
    s = list()
    # print(args)
    for c in args[1:]:
        # print(c)
        s.append(reverse_map[c[0]])
    name =  "_".join(sorted(s))
    print(name)
    
    return name
    
if __name__ == '__main__':
    sample_resizer_output()
