opt = {
    "model":"resizer", 
    "params":{
        "model_prefix":"resizerv43d_28_32_3res_", # the prefix used while training the model, 
        "epoch": 3, # the epoch for which the model needs to be evaluated, currently epoch 3 and 10
        "model_path":"/virat-vr/models/pytorch-i3d/", # directory where the models are saved (remember the trailing /)
        "data_root":"/mnt/data/TinyVIRAT/", # root folder to the tinyVIRAT dataset
        "classes_file": "classes.txt", 
        "model_type":"3d", # model type being evaluated, 2d or 3d
        "i3d_mode" : "32x112", # i3d mode same as in the i3d test files
        "num_frames": 32, #number of frames being given to the model at a time
        "num_workers": 6, # number of dataloader workers being used
        "batch_size": 4, # batch size of the evaluation block, will depend on the gpu 
        "resize_shape":28, # initial shape being fed to the resizer model
        "model_input_shape":112, # final shape being output by the resizer and being fed to the i3d model
        "num_resblocks": 3, # the number of residual blocks being used in the model
        "confusion":"confusion.npy", #the filename with which to save the confusion matrix for the test set 
        "predictions":"predictions.npy", #the filename with which to save thepreductions for the test set 
        "actuals":"actuals.npy", #the filename with which to save the actual labels for the test set 
        "logits":"logits.npy" #the filename with which to save the logits for the test set 
    }

}

def get():
    global opt
    return opt