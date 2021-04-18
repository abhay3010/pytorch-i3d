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


def run(init_lr=0.1, max_steps=64e3, mode='rgb',init_model='models/converted_i3d_rgb_charades.pt', root='/ssd/Charades_v1_rgb', classes_file="classes.txt",
 batch_size=8*5, save_model='', start_from=None):
    
    # setup dataset remember to change the crop size here.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_transforms = transforms.Compose([ videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(112)])

    dataset = Dataset(root, "train",classes_file, transforms=train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = Dataset(root, "test",classes_file, transforms=train_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    
    # setup the model
    i3d = InceptionI3d(157, in_channels=3)
    print(len(i3d.state_dict()))
    if start_from:
        i3d.replace_logits(26)
        i3d.load_state_dict(torch.load(start_from))
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
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])


    num_steps_per_update = 50 # accum gradient
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
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                print(inputs.shape)
                inputs = Variable(inputs.to(device))
                t = inputs.size(2)
                labels = Variable(labels.to(device))

                per_frame_logits = i3d(inputs)
                # upsample to input size
                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.item()

                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.item()

                loss = (0.01*loc_loss + 0.99*cls_loss)/num_steps_per_update
                tot_loss += loss.item()
                loss.backward()
                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    if steps % 10 == 0:
                        print ('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10))
                        # save model
                        torch.save(i3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if phase == 'val':
                print ('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter))
    


if __name__ == '__main__':
    # need to add argparse
    #run(mode=args.mode, root=args.root, save_model=args.save_model)
    root = "/virat-vr/TinyVIRAT/"
    max_steps = 320.0
    save_model='/virat-vr/models/pytorch-i3d/v1'
    run(root=root, max_steps=max_steps,save_model=save_model, batch_size=32 )
