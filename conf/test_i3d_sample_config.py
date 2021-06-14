opt = {
    "model":"i3d", 
    "params":{
        "model_prefix":"i3d_inp112_", # the prefix used while training the model, 
        "epoch_list": [1200], # the epochs for which the model needs to be evaluated, 
        "model_path":"/virat-vr/models/pytorch-i3d/", # directory where the models are saved (remember the trailing /)
        "data_root":"/mnt/data/TinyVIRAT/", # root folder to the tinyVIRAT dataset
        "i3d_mode" : "32x112",
        "num_frames": 32,
        "num_workers": 6,
        "batch_size": 12
    }

}

def get():
    global opt
    return opt