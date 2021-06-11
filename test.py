import pathlib
import sys
import posix
from pytorch_eval import eval_model_list as eval_i3d
from eval_resizer import eval_model_list as eval_resizer
from eval_resizer_spatial import eval_model_list as eval_spatial_resizer
import conf


def test(config):
    if config['model'] == 'i3d':
        eval_i3d(**config["params"])
    elif config['model'] == 'resizer':
        eval_resizer(**config['params'])
    elif config['model'] == 'spatial_resizer':
        eval_spatial_resizer(**config['params'])
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
    test(config)

if __name__ == '__main__':
    main()