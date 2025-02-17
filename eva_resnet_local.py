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
from torchsummary import summary
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np



def run(model, dataloader):
    predictions = list()
    trues = list()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for batch, labels in dataloader:
        num_videos = batch.size(0)
        batch = Variable(batch.to(device))
        with torch.set_grad_enabled(False):
            v = list()
            for i in range(num_videos):
                inputs = batch[i].permute(1,0,2,3)
                label = labels[i] 
                out = model(inputs)

                out = torch.sigmoid(out)
                y_p,_ = torch.max(out, dim=0)
                p = np.array([1 if z >=0.5 else 0 for z in y_p])
                predictions.append(p)
                trues.append(label.numpy())
        break    

    f1_macro = f1_score(trues, predictions, average='macro')
    f1_micro = f1_score(trues, predictions, average='micro')
    accuracy = accuracy_score(trues, predictions)
    

    print(f1_macro, f1_micro, accuracy)
    return f1_macro, f1_micro, accuracy

def eval_resnet(root, classes_file, model_path, batch_size, n_workers):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = models.resnet50()
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 26, bias=True)
    model_ft.load_state_dict(torch.load(model_path, map_location=device))
    model_ft.train(False)
    model_ft.to(device)
    if torch.cuda.device_count()>1:
        model_ft = nn.DataParallel(model_ft)
    dataset = Dataset(root, "test", classes_file, resize_shape=(224,224), transforms=videotransforms.Normalize([6.7193e-05, 6.9380e-05, 6.6246e-05], [0.0061, 0.0063, 0.0061]), sample=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    run(model_ft, dataloader)
def main():
    #Local Params
    root = "./TinyVIRAT"
    classes_file = "classes.txt"
    model_path = "eval_models/resnet50_v1000000.pt"
    batch_size=2
    n_workers=0

    #GPU Params
    # root = "./TinyVIRAT"
    # classes_file = "classes.txt"
    # model_path = "eval_models/resnet50_v1000000.pt"
    # batch_size=2
    # n_workers=0

    eval_resnet(root, classes_file, model_path, batch_size, n_workers)
def means():
    root = "./TinyVIRAT"
    classes_file = "classes.txt"
    model_path = "eval_models/resnet50_v1000000.pt"
    batch_size=2
    n_workers=0
    dataset = Dataset(root, "train", classes_file, resize=False)
    means = list()
    means_squared = list()
    v_sizes = list()
    count = 0
    for d, l in dataset:
        y = d.permute(1,2,3,0)
        vsize = y.size(0)
        m = torch.mean(y, dim=[0,1,2])
        means.append(m)
        means_squared.append(torch.mean(y**2, dim=[0,1,2]))
        v_sizes.append(vsize)
        count+=1
        if count%100 == 0:
            print(count)
    mean = torch.zeros((1,3))
    mean_squares = torch.zeros((1,3))
    s = sum(v_sizes)
    for m, m_sq, f in zip(means, means_squared, v_sizes):
        mean+= (m*1.0*f)/s
        mean_squares+= (m_sq*1.0*f)/s
    print("mean ", mean, "mean squares ", mean_squares)
    std =  (mean_squares - mean**2)**0.5
    print(mean, std)
    

    
if __name__ == '__main__':
    main()


