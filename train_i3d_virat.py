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


import numpy as np

from i3d import InceptionI3d

from virat_dataset import Virat as Dataset
from torchsummary import summary



def run(init_lr=0.1, max_steps=64e3,i3d_mode='32x112', num_frames=32, mode='rgb',init_model='models/converted_i3d_rgb_charades.pt', root='/ssd/Charades_v1_rgb', classes_file="classes.txt",
 batch_size=8*5, save_model='', start_from=None):
    
    # setup dataset remember to change the crop size here.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_transforms = transforms.Compose([ videotransforms.RandomHorizontalFlip(),
    ])

    dataset = Dataset(root, "train",classes_file, num_frames=num_frames, transforms=train_transforms, shuffle=True, downscale=True, downscale_shape=(28,28), sample=False)
    train, val = dataset.get_train_validation_split(test_perc=0.1)
    val_dataset = torch.utils.data.Subset(dataset, val)
    train_dataset = torch.utils.data.Subset(dataset, train)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    
    # setup the model
    i3d = InceptionI3d(157,mode=i3d_mode, in_channels=3)
    if start_from:
        state_dict = torch.load(start_from)
        i3d.replace_logits(26)
        new_dict = dict()
        for k, v in state_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        i3d.load_state_dict(new_dict)
    else:
        y = torch.load('models/rgb_charades.pt')
        y_new = dict()
        for k,v in y.items():
            if k not in ['logits.conv3d.weight', 'logits.conv3d.bias'] :
                y_new['model.'+k] = v
            else:
                y_new[k] = v
        i3d.load_state_dict(y_new)
        i3d.replace_logits(26)
    i3d.to(device)
    if torch.cuda.device_count()>1:
        i3d = nn.DataParallel(i3d)
    
    lr = init_lr
    optimizer = optim.Adam(i3d.parameters(), lr=lr)
    #lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [800, 1600, 3200, 6400])


    num_steps_per_update = 2 # accum gradient
    #num_steps_per_update = 1
    steps = 0
    # train it
    while steps < max_steps:#for epoch in range(num_epochs):
        print ('Step {}/{}'.format(steps, max_steps))
        print ('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode
                
            tot_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            for data in dataloaders[phase]:
                # print("Iter {0}".format(num_iter))
                num_iter += 1
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                # print(inputs.shape)
                inputs = Variable(inputs.to(device))
                t = inputs.size(2)
                labels = Variable(labels.to(device))

                per_video_logits = i3d(inputs)
                # upsample to input size
                #print(per_video_logits.shape, labels.shape)
                #print(per_frame_logits)
                #print(labels)
                #print(per_video_logits.shape)
                
                 

                # compute localization loss
                class_loss = F.binary_cross_entropy_with_logits(per_video_logits, labels)
                loss = class_loss/num_steps_per_update
                tot_loss += loss.item()
                loss.backward()
                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    # lr_sched.step()
                    if steps % 10 == 0:
                        print ('{} Loss: {:.4f}'.format(phase, tot_loss/10))
                        # save model                        
                        tot_loss  = 0.
                        if steps %100 == 0:
                            torch.save(i3d.state_dict(), save_model+str(steps).zfill(6)+'.pt')
            if phase == 'val':
                print ('{}  Loss: {:.4f} '.format(phase, (tot_loss*num_steps_per_update)/num_iter))
    

def main():
    #Gpu params
    # need to add argparse
    root = "/mnt/data/TinyVIRAT/"
    max_steps = 64000.0
    save_model='/virat-vr/models/pytorch-i3d/i3d_inp28_'

    #Local params
    # start_from = None
    # root = "TinyVIRAT/"
    # max_steps = 320000.0
    # save_model=''
    # start_from = None
    run(init_lr=0.0001, root=root, i3d_mode='32x112', num_frames=32, max_steps=max_steps,save_model=save_model, batch_size=12, start_from=start_from)

def mode_summary():
    model = i3d = InceptionI3d(26, in_channels=3)
    summary(model, (3,32,112,112), batch_size=2)

if __name__ == '__main__':
    main()
    