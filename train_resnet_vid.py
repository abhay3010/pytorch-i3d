import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms, models
import videotransforms
from virat_dataset import Virat as Dataset
from virat_dataset import collate_tensors
from torchsummary import summary
from resizer import ResizerWithTimeCompression
from collections import OrderedDict




def train_model(model, dataloaders, criterion, optimizer, model_prefix='', num_epochs=25  ):


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            counter = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':    
                        loss.backward()     
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                counter+=1
                if counter%100 == 0:
                    print("step ", counter, running_loss/(counter*inputs.size(0)))
            epoch_loss = running_loss /len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} '.format(phase, epoch_loss) )
        if phase == 'val':
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), model_prefix+str(epoch).zfill(6)+'.pt')
            else:
                torch.save(model.state_dict(), model_prefix  + str(epoch).zfill(6)+'.pt')

def run(root, classes_file,save_path, batch_size=2, lr=0.00005):
    #Initialise the dataset, loaders and model with the right set of parameters. 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_transforms = transforms.Compose([ 
        videotransforms.RandomHorizontalFlip(),
        videotransforms.Normalize([0.4719, 0.5126, 0.5077], [0.2090, 0.2103, 0.2152])

    ])
    dataset = Dataset(root, "train",classes_file, transforms=train_transforms, normalize=False, sample=True)
    training, test = dataset.get_train_validation_split()
    train_dataset = torch.utils.data.Subset(dataset, training)
    val_dataset = torch.utils.data.Subset(dataset, test)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_tensors)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_tensors)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    model_ft = models.resnet50(pretrained=True)
    model_resizer = ResizerWithTimeCompression(3, 1, 1, (224,224), squeeze=True)
    set_parameters_requires_grad(model_ft, True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 26, bias=True)
    final_model = nn.Sequential(OrderedDict([ ('resizer',model_resizer),
        ('resnet',model_ft)
    ]))
    params_to_update = []
    for name,param in final_model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
    optimizer_ft = optim.AdamW(params_to_update, lr=lr)
    ## define the criteria
    criterion  = nn.BCEWithLogitsLoss()
    final_model.to(device)
    if torch.cuda.device_count()>1:
        final_model = nn.DataParallel(final_model)
    train_model(final_model, dataloaders,criterion, optimizer_ft, model_prefix=save_path )


def set_parameters_requires_grad(model, deactivate):
    if deactivate:
        for param in model.parameters():
            param.requires_grad = False

def main():
    #Local parameters
    # root = "TinyVIRAT/"
    # classes_file =  "classes.txt"
    # save_path = ''


    #gpu paramaeters
    root = "/mnt/data/TinyVIRAT/"
    classes_file =  "classes.txt"
    save_path = '/virat-vr/models/pytorch-i3d/resnet50_video_v1'

    run(root,classes_file,save_path)
def test_dataset():
    root = "./TinyVIRAT"
    train_transforms = transforms.Compose([ 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4719, 0.5126, 0.5077], [0.2090, 0.2103, 0.2152])

    ])
    dataset = Dataset(root, "test","classes.txt", resize=True, resize_shape=(224,224), transforms=train_transforms, shuffle=False)
    train,test = dataset.get_train_test_split()
    print(len(train), len(test))

if __name__ == '__main__':
    main()

