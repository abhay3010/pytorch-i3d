import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
import videotransforms
from resnet_test_dataset import ViratResnetValidation as Dataset
from per_frame_image_dataset import ViratImages
from torchsummary import summary




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
                num_videos = inputs.size(0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    for c in range(num_videos):    
                        outputs = model(inputs[c].permute(1,0,2,3))
                        outputs = torch.mean(outputs, dim=0)
                        loss = criterion(outputs, labels[c])
                        if phase == 'train':    
                            loss.backward()     
                        running_loss += loss.item()
                    if phase == 'train':
                         optimizer.step()
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

def run(root, classes_file,save_path, batch_size=4, lr=0.001):
    #Initialise the dataset, loaders and model with the right set of parameters. 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_transforms = transforms.Compose([
        videotransforms.RandomHorizontalFlip(),
        videotransforms.Normalize([0.4719, 0.5126, 0.5077], [0.2090, 0.2103, 0.2152])])
   
    dataset = Dataset(root, "train", classes_file, resize_shape=(224,224), transforms=data_transforms, sample=False)
    train, test = dataset.get_train_validation_split()
    train_dataset = torch.utils.data.Subset(dataset, train)
    val_dataset = torch.utils.data.Subset(dataset, test)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    model_ft = models.resnet50(pretrained=True)
    set_parameters_requires_grad(model_ft, True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 26, bias=True)
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
    optimizer_ft = optim.Adam(params_to_update, lr=lr)
    ## define the criteria
    criterion  = nn.BCEWithLogitsLoss()
    model_ft.to(device)
    if torch.cuda.device_count()>1:
        model_ft = nn.DataParallel(model_ft)
    train_model(model_ft, dataloaders,criterion, optimizer_ft, model_prefix=save_path )


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
    save_path = '/virat-vr/models/pytorch-i3d/resnet50_batch_mean_ftune_lf_v5'

    run(root,classes_file,save_path, batch_size=8, lr=0.00005)
def test_dataset():
    root = "/mnt/data/TinyVIRAT/"
    train_transforms = transforms.Compose([ 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])
    dataset = Dataset(root, "test","classes.txt", resize=True, resize_shape=(224,224), transforms=train_transforms, shuffle=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    for img, label in dataloader:
        print(img.shape)
def print_resnet():
    model_ft = models.resnet50(pretrained=True)
    print(model_ft)
def get_histogram():
    root = "./TinyVirat"
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = ViratImages(root, "test","classes.txt", resize=True, resize_shape=(224,224), transforms=train_transforms, shuffle=False,sample=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    h = None
    for img, label in dataloader:
        print(img.shape)
        h = torch.histc(img, bins=100)
        print(torch.max(img).item())
        print(torch.min(img).item())
        print(h)
        break
    dataset = Dataset(root, "test","classes.txt", resize=True, resize_shape=(224,224), num_frames=1, transforms=videotransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), shuffle=False,sample=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    h_p = None
    for img, label in dataloader:
        print(img.shape)
        h_p = torch.histc(img, bins=100)
        print(torch.max(img).item())
        print(torch.min(img).item())
        print(h_p)
        break

def test_dataset():
    dataset = ViratImages("./TinyVIRAT", "train","classes.txt", resize=True, resize_shape=(224,224), shuffle=False)
    training, test = dataset.get_train_validation_split()
    print(len(training), len(test))
    train_dataset = torch.utils.data.Subset(dataset, training)
    test_dataset = torch.utils.data.Subset(dataset, test)
    print(len(train_dataset), len(test_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    print(print(val_dataloader))

if __name__ == '__main__':
    main()

