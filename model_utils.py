"""
    Utilities for loading and freezing the parameters of a model
"""
import torch
#Hacky: Will need to be fixed later
def load_params_from_file(model, params_file, device):
    state_dict = torch.load(params_file, map_location=device)
    new_dict = dict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_dict[k[7:]] = v
        else:
            new_dict[k] = v
    model.load_state_dict(new_dict)
    return model

        
