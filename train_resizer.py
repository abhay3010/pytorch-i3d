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
from model_utils import *
from virat_dataset import collate_tensors_min, collate_tensors 


import numpy as np

from i3d import InceptionI3d
from resizer import *

from virat_dataset import Virat as Dataset

def run(data_root, i3d_model_path, batch_size, num_frames=32, model_input_shape=112,  data_input_shape=56, save_path='', init_lr = 0.001 ,num_epochs=10,i3d_mode='32x112', classes_file='classes.txt', num_resblocks=1, model_type='2d', freeze_i3d=True, num_workers=4,     num_steps_per_update = 2    
):
    #load the virat model. Freeze its layers. (check how to do so)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #print("torch device",device)
    i3d = InceptionI3d(26,mode=i3d_mode, in_channels=3)
    #print("declared model")
    i3d = load_i3d_from_file(i3d, i3d_model_path, device, freeze_i3d)
    #load the resizer_model
    # resizer = BranchedResizerV2(3, int(v_mode.split('x')[0]), model_input_shape,num_resblocks=1)
    # resizer = ResizerMainNetworkV4_2D(3, int(v_mode.split('x')[0]), (model_input_shape, model_input_shape), num_resblocks=3)
    resizer = get_resizer_model(model_type, i3d_mode, model_input_shape, num_resblocks)
    #load the virat dataset
    train_transforms = transforms.Compose([ videotransforms.RandomHorizontalFlip(),
    ])
    dataset = Dataset(data_root, "train",classes_file,num_frames=num_frames, resize=True,resize_shape=(data_input_shape,data_input_shape), transforms=train_transforms,sample=False)
    train, test = dataset.get_train_validation_split()
    train_dataset = torch.utils.data.Subset(dataset, train)
    val_dataset = torch.utils.data.Subset(dataset, test)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,  shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,  shuffle=True, num_workers=num_workers, pin_memory=True)
    dataloaders = {'train': dataloader, 'val': val_dataloader}
    #Move both models to devices
    i3d.to(device)
    resizer.to(device)
    if torch.cuda.device_count()>1:
        i3d = nn.DataParallel(i3d)
        resizer = nn.DataParallel(resizer)
    lr = init_lr
    optimizer = optim.Adam(list(resizer.parameters()) + list(i3d.parameters()), lr=lr)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.0001, verbose=True)

    #print("resizer network", resizer)
    #print("i3d", i3d)
    for epoch in range(num_epochs):
        print ('Epoch {}/{}'.format(epoch, num_epochs))
        print ('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                resizer.train(True)
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode
                resizer.train(False)
                
            tot_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            i3d.zero_grad()
            resizer.zero_grad()
            for data in dataloaders[phase]:
                num_iter += 1
                inputs, labels = data

                # wrap them in Variable
                # print(inputs.shape)
                inputs = Variable(inputs.to(device))
                labels = Variable(labels.to(device))
                # print("original input shape", inputs.shape)
                resized_image = resizer(inputs)
                # print("resized input shape", resized_image.shape)
                per_video_logits = i3d(resized_image)
                class_loss = F.binary_cross_entropy_with_logits(per_video_logits, labels)
                loss = class_loss/num_steps_per_update
                tot_loss += loss.item()
                loss.backward()
                if num_iter == num_steps_per_update and phase == 'train':
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    i3d.zero_grad()
                    # lr_sched.step()
                    print ('{} Loss: {:.4f}'.format(phase, tot_loss))                    
                    tot_loss  = 0.
            if phase == 'val':
                print ('{}  Loss: {:.4f} '.format(phase, (tot_loss*num_steps_per_update)/num_iter))
                #scheduler.step((tot_loss*num_steps_per_update)/num_iter)
        if isinstance(resizer, nn.DataParallel):
            torch.save(resizer.module.state_dict(), save_path+str(epoch).zfill(6)+'.pt')
            torch.save(i3d.module.state_dict(), save_path + 'i3d' + str(epoch).zfill(6)+'.pt')
        else:
            torch.save(resizer.state_dict(), save_path+str(epoch).zfill(6)+'.pt' )
            torch.save(i3d.state_dict(), save_path + 'i3d' + str(epoch).zfill(6)+'.pt')

def main():
    # Local parameters
    # data_root = 'TinyVIRAT'
    # data_input_shape= (14, 14)
    # model_input_shape = (112, 112)
    # virat_model_path = '/workspaces/pytorch-i3d/eval_models/v5004080.pt'
    # batch_size = 2
    # save_model = 'bilinear_32_resizer_v1'

    #GPU parameters
    data_root = '/mnt/data/TinyVIRAT/'
    model_input_shape = 112
    i3d_model_path = '/virat-vr/models/pytorch-i3d/i3d_inp28_002400.pt'
    batch_size = 40
    save_model = '/virat-vr/models/pytorch-i3d/branched_resizer_v2_56_'
    data_input_shape = 56

    num_epochs=50
    run(data_root,i3d_model_path, batch_size, model_input_shape=model_input_shape,save_path=save_model, data_input_shape=data_input_shape, num_epochs=num_epochs)

def get_resizer_model(model_type, v_mode, model_input_shape, num_resblocks):
    if model_type == '2d':
        return ResizerMainNetworkV4_2D(3, int(v_mode.split('x')[0]), (model_input_shape, model_input_shape), num_resblocks=num_resblocks)
    else:
        return ResizerMainNetworkV4_3D(3, int(v_mode.split('x')[0]), (model_input_shape, model_input_shape), num_resblocks=num_resblocks )
    

if __name__ == '__main__':
    main()
