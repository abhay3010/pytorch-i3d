import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from collections import OrderedDict
import torchvision
from torchvision import datasets, transforms, models
import videotransforms
from virat_dataset import Virat as Dataset
from torchsummary import summary
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from resizer import ResizerWithTimeCompression
import numpy as np



def run(model, dataloader):
    predictions = list()
    trues = list()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for batch, labels in dataloader:
        batch = batch.to(device)
        with torch.set_grad_enabled(False):
            inputs = Variable(batch.to(device))
            
            v = torch.sigmoid(model(batch))
            for y_t, y_p in zip(labels, v):
                p = np.array([1 if z >=0.5 else 0 for z in y_p])
                predictions.append(p)
                trues.append(y_t.numpy())    

    f1_macro = f1_score(trues, predictions, average='macro')
    f1_micro = f1_score(trues, predictions, average='micro')
    accuracy = accuracy_score(trues, predictions)
    

    print(f1_macro, f1_micro, accuracy)
    return f1_macro, f1_micro, accuracy

def eval_resnet(root, classes_file, model_path, batch_size, n_workers):
    model_ft = models.resnet50(pretrained=True)
    model_resizer = ResizerWithTimeCompression(3, 1, 1, (224,224), squeeze=True)
    final_model = nn.Sequential(OrderedDict([ ('resizer',model_resizer),
        ('resnet',model_ft)
    ]))
    final_model.load_state_dict(torch.load(model_path))
    if torch.cuda.device_count()>1:
        final_model = nn.DataParallel(final_model)
    dataset = Dataset(root, "test", classes_file, transforms=videotransforms.Normalize([0.4719, 0.5126, 0.5077], [0.2090, 0.2103, 0.2152]), normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,num_workers=n_workers, pin_memory=True)
    f1_macro, f1_micro, accuracy = run(final_model, dataloader)
    return f1_macro, f1_micro, accuracy
def main():
    #Local Params
    # root = "./TinyVIRAT"
    # classes_file = "classes.txt"
    # model_path = "eval_models/resnet50_v1000000.pt"
    # batch_size=2
    # n_workers=0

    #GPU Params
    root = "/mnt/data/TinyVIRAT"
    classes_file = "classes.txt"
    batch_size=2
    n_workers=4
    models = list()
    model_prefix = "resnet50_video_v1"
    for epoch in range(1,25):
        m = model_prefix + str(epoch).zfill(6)+'.pt'
        models.append(m)
    base_path = '/virat-vr/models/pytorch-i3d/'
    for model in models:
        f1_macro, f1_micro, accuracy = eval_resnet(root, classes_file, base_path+model, batch_size, n_workers)
        print("{0}: f1_macro {1}, f1_micro: {2}, accuracy: {3}".format(model, f1_macro, f1_micro, accuracy))