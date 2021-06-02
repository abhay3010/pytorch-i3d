import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms, models
import videotransforms
from per_frame_image_dataset import ViratImages as Dataset
from torchsummary import summary
from two_stream_i3d import TwoStreamNetwork
from i3d import InceptionI3d
from resizer import ResizerMainNetworkV4_2D, ResizerMainNetworkV4_3D
from spatial_transformer import *




def train_model(model, dataloaders, criterion, optimizer,model_prefix='', num_epochs=25  ):
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

def run(root, classes_file, save_path, resizer_path, i3d_path,v_mode='32x112',model_input_shape=(112,112), batch_size=256, lr=0.001):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #Initialise the dataset, loaders and model with the right set of parameters. 
    train_transforms = transforms.Compose([ videotransforms.RandomHorizontalFlip(),
    ])
    dataset = Dataset(root, "train",classes_file,resize=True,resize_shape=(28,28), transforms=train_transforms,sample=True)
    training, test = dataset.get_train_validation_split()
    train_dataset = torch.utils.data.Subset(dataset, training)
    val_dataset = torch.utils.data.Subset(dataset, test)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)    
    dataloaders = {"train":dataloader, 'val':val_dataloader}
    helper_model = load_resizer_and_i3d(resizer_path, i3d_path, device)
    actual_model = load_actual_model(v_mode, model_input_shape)
    model_ft = TwoStreamNetwork(actual_model, helper_model, 26)
    optimizer_ft = optim.Adam(model_ft.get_parameters_to_train(), lr=lr)
    ## define the criteria
    criterion  = nn.BCEWithLogitsLoss()
    model_ft.to(device)
    if torch.cuda.device_count()>1:
        model_ft = nn.DataParallel(model_ft)
    train_model(model_ft, dataloaders,criterion, optimizer_ft, model_prefix=save_path )

def load_resizer_and_i3d(resizer_path, i3d_path, device):
    i3d = InceptionI3d(26, mode="32x112", in_channels=3)
    i3d.load_state_dict(torch.load(i3d_path, device))
    resizer = ResizerMainNetworkV4_3D(3, 32, (112,112),num_resblocks=1)
    resizer.load_state_dict(torch.load(resizer_path, device))
    return nn.Sequential(resizer, i3d)


def load_actual_model(v_mode, model_input_shape, input_res=28):
    i3d = InceptionI3d(157,mode="32x112", in_channels=3)
    y = torch.load('models/rgb_charades.pt')
    y_new = dict()
    for k,v in y.items():
        if k not in ['logits.conv3d.weight', 'logits.conv3d.bias'] :
            y_new['model.'+k] = v
        else:
            y_new[k] = v
    i3d.load_state_dict(y_new)
    i3d.replace_logits(26)
    model = nn.Sequential(
        SpatialTransformer(3, in_time=int(v_mode.split('x')[0]), in_res=input_res),
        ResizerMainNetworkV4_2D(3, int(v_mode.split('x')[0]), model_input_shape,num_resblocks=1),
        i3d)
    return model
def set_parameters_requires_grad(model, deactivate):
    if deactivate:
        for param in model.parameters():
            param.requires_grad = False

def main():
    #Local parameters
    # root = "TinyVIRAT/"
    # classes_file =  "classes.txt"
    # save_path = ''
    # resizer_path = 'eval_models/resizerv43d_28_32_000015.pt'
    # i3d_path = 'eval_models/resizerv43d_28_32_i3d000015.pt'
    # batch_size = 1
    #gpu paramaeters
    root = "/mnt/data/TinyVIRAT/"
    classes_file =  "classes.txt"
    save_path = '/virat-vr/models/pytorch-i3d/two_stream_28_deactivated'
    resizer_path = '/virat-vr/models/pytorch-i3d/resizerv43d_28_32_000015.pt'
    i3d_path = '/virat-vr/models/pytorch-i3d/resizerv43d_28_32_i3d000015.pt'
    batch_size = 12


    run(root,classes_file, save_path, resizer_path,i3d_path, batch_size=batch_size)
if __name__ == '__main__':
    main()

