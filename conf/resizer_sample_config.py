opt = {
    "model":'resizer',
    "resizer_type":"2d", #type of resizer, whether 2d or 3d. 
    "params":{
    "data_root":"/mnt/data/TinyVIRAT/", # root folder to the tinyVIRAT dataset, 
    "save_path": "/virat-vr/models/pytorch-i3d/sample_resizer_model_", # prefix of the model being trained. Saved after every epoch, with the epoch id added 
    # "save_path": "virat_vr/sample_model_",
    # "data_root": "TinyVIRAT/",
    "num_epochs": 50, #number of epochs to train for
    "num_frames":32, #number of frames the model takes as input
    "init_lr":0.001, #initial learning rate, 
    "classes_file":"classes.txt", # the map of classes to indices used by the Dataset to translate the classes to 1-hot vectors
    "i3d_mode":"32x112", #i3d config num_framesXinput_resolution can vary as (16,32,64)x(112,224)
    "model_input_shape":112, # the shape that the i3d model expects from the resizer
    "data_input_shape":56, # the input resolution (data_input_shape x data_input_shape) that the resizer gets from the data loader
    "num_resblocks":1, # number of residual blocks in the resizer
    "i3d_model_path": "/virat-vr/models/pytorch-i3d/v7_bilinear_32_112004400.pt", # the initial model path for i3d
    "batch_size":32, # the batch size to be used 
    "freeze_i3d":True, # set to True if we need to freeze all but the last layers of the i3d model, False if you want to train i3d with the resizer
    "num_workers":4 # the number of processes that the dataloader needs to spawn
    

    }
}
def get():
    global opt
    return opt