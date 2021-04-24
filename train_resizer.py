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
from model_utils import load_params_from_file


import numpy as np

from i3d import InceptionI3d
from resizer import ResizerMainNetwork

from virat_dataset import Virat as Dataset

def run(data_root,data_input_shape, model_input_shape, virat_model_path,batch_size,save_model='', init_lr = 0.01 ,num_epochs=10,v_mode='32x112', classes_file='classes.txt'):
    #load the virat model. Freeze its layers. (check how to do so)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    i3d = InceptionI3d(26,mode=v_mode, in_channels=3)
    i3d = load_params_from_file(i3d, virat_model_path, device)
    #load the resizer_model
    resizer = ResizerMainNetwork(3, int(v_mode.split('x')[0]), model_input_shape)
    #load the virat dataset
    train_transforms = transforms.Compose([ videotransforms.RandomHorizontalFlip(),
    ])
    dataset = Dataset(data_root, "train",classes_file,resize_shape=data_input_shape, transforms=train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  shuffle=True, num_workers=0, pin_memory=True)

    val_dataset = Dataset(data_root, "test",classes_file,resize_shape=data_input_shape, transforms=None)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,  shuffle=True, num_workers=0, pin_memory=True)
    dataloaders = {'train': dataloader, 'val': val_dataloader}
    #Move both models to devices
    i3d.to(device)
    resizer.to(device)
    if torch.cuda.device_count()>1:
        i3d = nn.DataParallel(i3d)
        resizer = nn.DataParallel(resizer)
    lr = init_lr
    num_steps_per_update = 10
    #Only passing the resizer parameters to the optimizer
    optimizer = optim.Adam(resizer.parameters(), lr=lr)   
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
                t = inputs.size(2)
                labels = Variable(labels.to(device))
                print("original input shape", inputs.shape)
                resized_image = resizer(inputs)
                print("resized input shape", resized_image.shape)
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
        torch.save(i3d.state_dict(), save_model+str(epoch).zfill(6)+'.pt')

def main():
    # Local parameters
    # data_root = 'TinyVIRAT'
    # data_input_shape= (14, 14)
    # model_input_shape = (112, 112)
    # virat_model_path = '/workspaces/pytorch-i3d/eval_models/v5004080.pt'
    # batch_size = 1
    # save_model = 'bilinear_32_resizer_v1'

    #GPU parameters
    data_root = '/mnt/data/TinyVIRAT/'
    data_input_shape= (14, 14)
    model_input_shape = (112, 112)
    virat_model_path = 'v7_bilinear_32_112004400.pt'
    batch_size = 4
    save_model = 'bilinear_32_resizer_v1'
    num_epochs=50
    run(data_root, data_input_shape, model_input_shape, virat_model_path, batch_size, save_model, num_epochs=num_epochs)

if __name__ == '__main__':
    main()
