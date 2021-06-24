opt = {
    "model":"spatial_resizer", 
    "params":{
        "model_prefix":"resizer_spatial_1_5_3res_", # the prefix used while training the model, 
        "epoch_list": range(0,30), # the epochs for which the model needs to be evaluated, currently epoch 3 and 10
        "model_path":"/virat-vr-r-models/", # directory where the models are saved (remember the trailing /)
        "data_root":"/mnt/data/TinyVIRAT/", # root folder to the tinyVIRAT dataset
        "classes_file": "classes.txt", 
        "i3d_mode" : "32x112", # i3d mode same as in the i3d test files
        "num_frames": 32, #number of frames being given to the model at a time
        "num_workers": 4, # number of dataloader workers being used
        "batch_size": 64, # batch size of the evaluation block, will depend on the gpu 
        "resize_shape":56, # initial shape being fed to the resizer model
        "model_input_shape":112, # final shape being output by the resizer and being fed to the i3d model
        "num_resblocks": 3, # the number of residual blocks being used in the model
        "confusion":"_corrected_after_1_5.npy",
        "mode":"2d",
        "read_at":1,
        "apply_at":5

    }

}

def get():
    global opt
    return opt