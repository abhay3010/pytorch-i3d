import torch.utils.data as data_util
from torch.utils.data.dataloader import default_collate
import numpy as np
import json
import random
from pathlib import Path
import cv2
import os
import torch
import torch.nn.functional as F
from collections import defaultdict

#from joblib import Parallel, delayed
def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))
def load_from_frames(frame_root,start_frame,num_frames, resize, resize_shape, normalize, downscale, downscale_shape):
    fnames = []
    for i in range(start_frame, start_frame + num_frames):
            fnames.append(os.path.join(frame_root, 'frame_{0}.jpg'.format(i)))
    fnames = sorted(fnames)
    frames = []
    for fn in fnames:
        img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        if  img is None:
            print("Failed to read file {0}, will exit", fn)
        if downscale and (img.shape[0], img.shape[1]) != downscale_shape:
            #print("downscaling")
            img = cv2.resize(img, downscale_shape, interpolation=cv2.INTER_LINEAR)
            #print("downscaling", img.shape)
        if resize and (img.shape[0], img.shape[1]) != resize_shape:
            #this will need to change for the final results
            img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
        if normalize:
            img = (img/255.)*2 - 1
        else:
            img = img/255.
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)
    
def make_dataset(root, data_type,num_frames, labels_file, load_all):
    _root  = Path(root)
    type_to_file = {"train":"tiny_train.json", "test":"tiny_test.json"}
    dataset = None
    labels_map = None
    with open(_root.joinpath(type_to_file[data_type]), 'r+') as f:
        dataset = json.load(f)['tubes']
    with open(_root.joinpath(labels_file), 'r+') as f:
        labels_map = {v:k for k,v in enumerate([k[:-1] for k in f.readlines()])}
        print("label map", labels_map)
    print(len(dataset), len(labels_map))
    if not load_all:
        processed_dataset = [
            {'path':str(_root.joinpath("videos", data_type, v['path'])),
            'frames':v['dim'][0], 
            'label': v['label'],
            'dims': v['dim']
            } for v in dataset if v['dim'][0] >=num_frames]
    else:
        processed_dataset = [
            {'path':str(_root.joinpath("videos", data_type, v['path'])),
            'frames':v['dim'][0], 
            'label': v['label'],
            'dims': v['dim']
            } for v in dataset]

    
    return processed_dataset, labels_map
    
def load_rgb_frames(root_path,start_frame, num_frames,total_frames,  resize=False, resize_shape=(112, 112), normalize=True, downscale=False, downscale_shape=(14,14)):
    vpath = Path(root_path)
    parent_path = vpath.parents[0]
    #look for the frames filepath
    frames_folder = parent_path.joinpath(vpath.stem + '_frames')
    if frames_folder.exists() and frames_folder.is_dir() and len(get_frames(str(frames_folder))) == total_frames :
        #print("loading from existing frames")
        array_from_frames = load_from_frames(str(frames_folder),start_frame, num_frames, resize, resize_shape, normalize, downscale, downscale_shape)
        return array_from_frames
    else:
        if not vpath.exists():
            raise ValueError("filepath not found {0}".format(root_path))
        frames_folder.mkdir(exist_ok=True)
        cap = cv2.VideoCapture(str(vpath))
        count = 0
        save_path = frames_folder
        frames = list()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                fpath = save_path.joinpath("frame_{0}.jpg".format(count))
                cv2.imwrite(str(fpath),frame)
                count+=1
                frames.append(frame)
        finally:
            cap.release()
        saved_frames = load_from_frames(str(frames_folder),start_frame,num_frames, resize, resize_shape, normalize, downscale, downscale_shape)
        return saved_frames

def get_frames(p):
    y = list()
    for x in os.listdir(p):
        if x.endswith('.jpg'):
            y.append(x)
    return y


class Virat(data_util.Dataset):
    def __init__(self, root, dtype,labels_file,num_frames=32,resize=True, resize_shape=(112,112), downscale=False, downscale_shape=(14,14), shuffle=False, normalize=True, transforms=None,sample=False,load_all=False):
        self.root = root
        self.dtype = dtype
        self.labels_file = labels_file
        self.resize = resize
        self.resize_shape = resize_shape
        self.transforms = transforms
        self.num_frames = num_frames
        self.shuffule = shuffle
        self.normalize = normalize
        self.load_all = load_all
        self.downscale = downscale
        self.downscale_shape = downscale_shape
        if load_all and shuffle:
            raise ValueError("shuffle and load_all cannot both be true")
        self.data, self.labels_map = make_dataset(root, dtype,num_frames, labels_file, self.load_all)
        if sample:
            self.data = self.data[:50]
    
    def __getitem__(self, index):
        details = self.data[index]
        start_f = 0
        num_frames = self.num_frames
        if self.load_all:
            num_frames = min(details['frames'], self.num_frames)

        if self.shuffule:
            try:
                start_f = random.randint(0,details['frames']-self.num_frames-1)
            except:
                print("error while getting a start frame ", details)
    
        imgs = load_rgb_frames(details['path'],start_f, num_frames,details['frames'],self.resize, self.resize_shape, self.normalize, self.downscale, self.downscale_shape)
        if self.transforms:
            imgs = self.transforms(imgs)
        y = np.zeros(len(self.labels_map), dtype=np.float32)
        for c in details['label']:
            y[self.labels_map[c]] = 1
        Y = np.array(y)
        # print(imgs.shape, Y.shape)
        return video_to_tensor(imgs), torch.from_numpy(Y)

    def __len__(self):
        return len(self.data)
        
    def get_train_validation_split(self, test_perc=0.1):
        random.seed(22)
        labels_count = defaultdict(set)
        for i, d in enumerate(self.data):
            for k in d['label']:
                labels_count[k].add(i)
        test_samples = set()
        for k in labels_count.keys():
            c =  max(int(len(labels_count[k])*test_perc), 1)
            test_samples.update(random.sample(labels_count[k],c))
        all_indices = set(range(len(self.data)))
        training = all_indices - test_samples
        return list(training), list(test_samples)

    def __len__(self):
        return len(self.data)

    
def test_interpolate():
    frame_path = 'TinyVIRAT/videos/train/VIRAT_S_000203_07_001341_001458/0.mp4'
    total_frames = 77
    shape = (70,70)
    unresized_frames = load_rgb_frames(frame_path, 0, total_frames, total_frames, resize=False, normalize=False)
    print(unresized_frames.shape)
    resized_frames = load_rgb_frames(frame_path, 0, total_frames, total_frames, resize=True, resize_shape=(112, 112), normalize=False).transpose([3, 0, 1, 2])
    torch_frames = torch.from_numpy(unresized_frames.transpose([3, 0, 1, 2]))
    print("torch unresized frames", torch_frames.shape)
    torch_resized_frames = F.interpolate(torch_frames.unsqueeze(0),size=(77,112,112), mode='trilinear', align_corners = False).numpy().squeeze()
    print(resized_frames.shape, torch_resized_frames.shape)
    print(np.all(np.abs(resized_frames - torch_resized_frames) < 0.0005))


def collate_tensors(tensor_list):
    #tensors have shape CxTxHxW
    d0 = max([t.shape[1] for t,_ in tensor_list])
    d1 = max([t.shape[2] for t,_ in tensor_list ])
    d2 = max([t.shape[3] for t, _ in tensor_list])
    tensor_list = [(resize_video(t, (d0, d1, d2)) , l) if (t.shape[2], t.shape[3]) != (d1,d2) else (t,l) for t,l in tensor_list]
    return default_collate(tensor_list)

def collate_tensors_min(tensor_list):
    #tensors have shape CxTxHxW
    d0 = min([t.shape[1] for t,_ in tensor_list])
    d1 = min([t.shape[2] for t,_ in tensor_list ])
    d2 = min([t.shape[3] for t, _ in tensor_list])
    tensor_list = [(resize_video(t, (d0, d1, d2)) , l) if (t.shape[2], t.shape[3]) != (d1,d2) else (t,l) for t,l in tensor_list]
    return default_collate(tensor_list)

def collate_with_time(tensor_list):
    d0 = max([t.shape[1] for t,_ in tensor_list])
    d1 = max([t.shape[2] for t,_ in tensor_list ])
    d2 = max([t.shape[3] for t, _ in tensor_list])
    tensor_list = [(resize_video(t, (d0, d1, d2)) , l) if (t.shape[1], t.shape[2], t.shape[3]) != (d0,d1,d2) else (t,l) for t,l in tensor_list]
    return default_collate(tensor_list)
    
def resize_video(t, shape):
    return F.interpolate(t.unsqueeze(0),size=shape, mode='trilinear', align_corners = False).squeeze(0)


if __name__ == '__main__':
    test_interpolate()





