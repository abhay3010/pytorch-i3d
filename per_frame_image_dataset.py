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
from PIL import Image


#from joblib import Parallel, delayed
def img_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1]))
def load_frame(frame_root,frame_num,resize, resize_shape, normalize):
    fn = os.path.join(frame_root, 'frame_{0}.jpg'.format(frame_num))
    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    # print("loaded image", type(img))
    if  img is None:
        print("Failed to read file {0}, will exit", fn)
    if resize and (img.shape[0], img.shape[1]) != resize_shape:
        #this will need to change for the final results
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(img)
    
def make_dataset(root, data_type, labels_file):
    _root  = Path(root)
    type_to_file = {"train":"tiny_train.json", "test":"tiny_test.json"}
    dataset = None
    labels_map = None
    with open(_root.joinpath(type_to_file[data_type]), 'r+') as f:
        dataset = json.load(f)['tubes']
    with open(_root.joinpath(labels_file), 'r+') as f:
        labels_map = {v:k for k,v in enumerate([k[:-1] for k in f.readlines()])}
        # print("label map", labels_map)
    # processed_dataset = [
    #     {'path':str(_root.joinpath("videos", data_type, v['path'])),
    #     'frames':v['dim'][0], 
    #     'label': v['label'],
    #     'dims': v['dim']
    #     } for v in dataset if v['dim'][0] >=num_frames]
    processed_dataset = list()
    for d in dataset:
        for i in range(d['dim'][0]):
            processed_dataset.append(
                {'path':str(_root.joinpath("videos", data_type, d['path'])),
                'frame_num':i,
                'label': d['label'],
                'dims':d['dim'],
                'frames':d['dim'][0]
                }
            )

    
    print(len(processed_dataset), len(labels_map))
    return processed_dataset, labels_map
    
def load_rgb_frame(root_path,start_frame, total_frames, resize=False, resize_shape=(112, 112), normalize=True):
    vpath = Path(root_path)
    parent_path = vpath.parents[0]
    #look for the frames filepath
    frames_folder = parent_path.joinpath(vpath.stem + '_frames')
    if frames_folder.exists() and frames_folder.is_dir() and len(get_frames(str(frames_folder))) == total_frames :
        #print("loading from existing frames")
        array_from_frames = load_frame(str(frames_folder),start_frame, resize, resize_shape, normalize)
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
        saved_frames = load_frame(str(frames_folder),start_frame,resize, resize_shape, normalize)
        return saved_frames

def get_frames(p):
    y = list()
    for x in os.listdir(p):
        if x.endswith('.jpg'):
            y.append(x)
    return y


class ViratImages(data_util.Dataset):
    def __init__(self, root, dtype,labels_file,resize=True, resize_shape=(112,112), shuffle=False, normalize=True, transforms=None,sample=False):
        self.root = root
        self.dtype = dtype
        self.labels_file = labels_file
        self.resize = resize
        self.resize_shape = resize_shape
        self.transforms = transforms
        self.shuffule = shuffle
        self.normalize = normalize
        self.data, self.labels_map = make_dataset(root, dtype, labels_file)
        if sample:
            self.data = self.data[:50]
    
    def __getitem__(self, index):
        details = self.data[index]
        start_f = details['frame_num']
        img = load_rgb_frame(details['path'],start_f,details['frames'],self.resize, self.resize_shape, self.normalize)
        # print(type(img))
        if self.transforms:
            img = self.transforms(img)
        y = np.zeros(len(self.labels_map), dtype=np.float32)
        for c in details['label']:
            y[self.labels_map[c]] = 1
        Y = np.array(y)
        # print(img.shape, Y.shape)
        # print(type(img))
        return img, torch.from_numpy(Y)

    def __len__(self):
        return len(self.data)

def collate_tensors(tensor_list):
    #tensors have shape CxTxHxW
    d0 = max([t.shape[1] for t,_ in tensor_list])
    d1 = max([t.shape[2] for t,_ in tensor_list ])
    d2 = max([t.shape[3] for t, _ in tensor_list])
    tensor_list = [(resize_video(t, (d0, d1, d2)) , l) if (t.shape[2], t.shape[3]) != (d1,d2) else (t,l) for t,l in tensor_list]
    return default_collate(tensor_list)
    
def resize_video(t, shape):
    return F.interpolate(t.unsqueeze(0),size=shape, mode='trilinear', align_corners = False).squeeze(0)