opt = {
    "model":'spatial_resizer',
    "params":{
    "data_root":"/mnt/data/TinyVIRAT/", # root folder to the tinyVIRAT dataset, 
    "save_path": "/virat-vr-r-models/resizer_spatial_28_1_8_3res_", # prefix of the model being trained. Saved after every epoch, with the epoch id added 
    # "save_path": "virat_vr/sample_model_",
    # "data_root": "TinyVIRAT/",
    "num_epochs": 30, #number of epochs to train for
    "num_frames":32, #number of frames the model takes as input
    "init_lr":0.0015, #initial learning rate, 
    "classes_file":"classes.txt", # the map of classes to indices used by the Dataset to translate the classes to 1-hot vectors
    "i3d_mode":"32x112", #i3d config num_framesxinput_resolution can vary as (16,32,64)x(112,224)
    "model_input_shape":112, # the shape that the i3d model expects from the resizer
    "data_input_shape":28, # the input resolution (data_input_shape x data_input_shape) that the resizer gets from the data loader
    "num_resblocks":3, # number of residual blocks in the resizer
    "i3d_model_path": "/virat-vr/models/pytorch-i3d/v7_bilinear_32_112004400.pt", # the initial model path for i3d
    "batch_size":32, # the batch size to be used 
    "freeze_i3d":False, # set to True if we need to freeze all but the last layers of the i3d model
    "num_workers":5, # the number of processes that the dataloader needs to spawn, 
    "num_steps_per_update":8,
    "mode":'2d',
    "read_at":1, 
    "apply_at":8
    }
}
def get():
    global opt
    return opt