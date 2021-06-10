opt = {
    "model":'i3d',
    "params":{
    # "data_root":"/mnt/data/TinyVIRAT/", # root folder to the tinyVIRAT dataset, 
    # "save_path": "/virat-vr/models/pytorch-i3d/sample_model_", # prefix of the model being trained. Saved after every epoch, with the epoch id added 
    "save_path": "virat_vr/sample_model_",
    "data_root": "TinyVIRAT/",
    "max_steps": 64000, #Steps to train for
    "num_frames":32, #number of frames the model takes as input
    "init_lr":0.001, #initial learning rate, 
    "classes_file":"classes.txt", # the map of classes to indices used by the Dataset to translate the classes to 1-hot vectors
    "i3d_mode":"32x112"
   
    }
}
def get():
    global opt
    return opt