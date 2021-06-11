import pathlib
import sys
import posix

from torch._C import Value
from train_i3d_virat import run as train_i3d
from train_resizer import run as train_resizer
from train_resizer_spatial import run as train_spatial_resizer
import conf
def train(config):
    validate_config(config)
    if config['model'] == 'i3d':
        train_i3d(**config["params"])
    elif config['model'] == 'resizer':
        train_resizer(**config["params"])
    elif config['model'] == 'spatial_resizer':
        train_spatial_resizer(**config["params"] )
    else:
        ValueError("Unknown model type {0}".format(config['model']))




def main():
    config = dict()
    if len(sys.argv) != 2:
        print("Correct usage : python train.py <config_file>")
        sys.exit("Invalid usage")
    try:
        config = conf.parse(sys.argv[1])
    except Exception as e:
        print("Unable to load the config",e )
        sys.exit()
    if 'model' not in config or config['model'] not in ['i3d', 'resizer', 'spatial_resizer']:
        print("model not in config, provide the model to train")
        sys.exit()
    train(config)

def validate_config(config):
    common_required_params = ['data_root', 'save_path']
    for c in common_required_params:
        if c not in config["params"]:
            raise ValueError("Key {0} missing in config".format(c))
    build_dir(config['params']['save_path'])
    if config['model'] == 'i3d':
        return

    elif config['model'] in ['resizer', 'spatial_resizer'] :
        resizer_required_params = ['model_input_shape', 'data_input_shape', 'num_resblocks', 'i3d_model_path']
        for c in resizer_required_params:
            if c not in config['params']:
                raise ValueError("Key {0} missing in config".format(c))

        return
    else:
        raise ValueError("Unrecognized model {0}".format(config['model']))
    

def build_dir(save_prefix):
    p = pathlib.Path(save_prefix).parent
    p.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    main()





    