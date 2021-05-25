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
from resizer import *

from virat_dataset import Virat as Dataset
from torchsummary import summary
from virat_dataset import collate_tensors, load_rgb_frames

def eval(resizer_model, model_path, root, classes_file, debug=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    val_dataset = Dataset(root, "test",classes_file,num_frames=32, resize=False, transforms=None)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_tensors) 
    resizer = ResizerMainNetworkV4_3D(3, 32, (112, 112), skip=False, num_resblocks=2)
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

    print(f1_macro, f1_micro, accuracy)
    if debug:
        pred_np = np.asarray(predictions)
        act_np = np.asarray(trues)
        cf = multilabel_confusion_matrix(trues, predictions)
        np.save('predictions.npy', pred_np)
        np.save('actuals.npy', act_np)
        np.save('logits.npy', np.asarray(p_logits))
        np.save('confusion.npy', cf)
        val_dataset = Dataset(root, "train",classes_file,num_frames=32, resize=False, transforms=None)
        _,val = val_dataset.get_train_validation_split(0.3)
        subset = torch.utils.data.Subset(val_dataset, val)
        val_loader = torch.utils.data.DataLoader(subset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_tensors)
        predictions = list()
        trues = list()
        p_logits = list()
        for batch, labels in val_loader:
            count+=1
            inputs = Variable(batch.to(device))
            out = resizer(inputs)
            v = torch.sigmoid(i3d(out))
            for y_t, y_p in zip(labels, v):
                p = np.array([1 if z >=0.5 else 0 for z in y_p])
                predictions.append(p)
                trues.append(y_t.numpy()) 
                p_logits.append(y_p.cpu().detach().numpy())
        pred_np = np.asarray(predictions)
        act_np = np.asarray(trues) 
        np.save('val_predictions.npy', pred_np)
        np.save('val_actuals.npy', act_np)


    return f1_macro, f1_micro, accuracy

def main():
    #i3d_model = "/virat-vr/models/pytorch-i3d/v7_bilinear_32_112004400.pt"
    prefix = 'branched_resizer_v2_32_112'
    model_list = list()
    for epoch in range(5,28):
        model_list.append((prefix+str(epoch).zfill(6)+'.pt', prefix+ 'i3d'+str(epoch).zfill(6)+'.pt'))
    for model, i3d_model in model_list:
       f1_macro, f1_micro, accuracy = eval('/virat-vr/models/pytorch-i3d/'+ model, '/virat-vr/models/pytorch-i3d/'+ i3d_model, "/mnt/data/TinyVIRAT/", "classes.txt", debug=True)
       print ("{0} , f1_macro : {1}, f1_micro {2}, Accuracy {3}".format(model,f1_macro, f1_micro, accuracy))


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
    resizer_model = 'eval_models/bilinear_32_resizer_v9_final_resizer_v43r_residuals_000038.pt'
    val_dataset = Dataset(root, "test",classes_file, resize=False, transforms=None, sample=True)
    x = val_dataset[3]
    print(x[0].shape)
    return
    new_val_dataset = Dataset(root,"test", classes_file, resize=True, resize_shape=(112,112), transforms=None, sample=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_tensors)
    resizer =ResizerMainNetworkV4_3D(3, 32, (112, 112), skip=False, num_resblocks=2)
    resizer_skip = ResizerMainNetworkV2(3,32,(112,112), skip=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resizer.load_state_dict(torch.load(resizer_model, map_location=device))
    resizer.to(device)
    resizer_skip.to(device)
    index = 0
    for batch, label in val_dataloader:

        resized_image_sp = resizer(batch).squeeze(0)
        print("resizer shape", resized_image_sp.shape)
        resized_normal = new_val_dataset[index][0]
        print("resized normal shape", resized_normal.shape)
        permuted_view = (resized_image_sp.permute(1,0,2,3) + 1)/2
        permuted_view_n = (resized_normal.permute(1,0,2,3) +1)/2
        print(permuted_view.size(0))
        for i in range(permuted_view.size(0)):
            save_image(permuted_view[i], "resized_frames/test_{0}_frame{1}_sp.png".format(index, i))
            save_image(permuted_view_n[i], "resized_frames/test_{0}_frame{1}_normal.png".format(index, i))
        index+=1
    







if __name__ == '__main__':
    main()
