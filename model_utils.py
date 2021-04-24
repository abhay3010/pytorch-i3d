"""
    Utilities for loading and freezing the parameters of a model
"""
from collections import OrderedDict
import torch
#Hacky: Will need to be fixed later
def load_params_from_file(model, params_file, device):
    print("loading", params_file)
    state_dict = torch.load(params_file, map_location=device)
    
    new_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_dict[k[7:]] = v
        else:
            print(k)
            new_dict[k] = v
    model.load_state_dict(new_dict)
    return model

def load_module_params_from_file(model, params_file, device):
    print("loading", params_file)
    state_dict = torch.load(params_file)
    
    new_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_dict[k[7:]] = v
        else:
            print(k)
            new_dict[k] = v
    model.load_state_dict(new_dict)
    return model
        
