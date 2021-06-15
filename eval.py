import pathlib
import sys
import posix
from eval_resizer import eval_model_list as eval_resizer
from eval_resizer_spatial import eval_model_list as eval_spatial_resizer
import conf


def eval(config):
    config['params']['debug'] = True
    config['params']['epoch_list'] = [config['params']['epoch']]
    del config['params']['epoch']
    for key in ['confusion_file', 'predictions_file', 'actuals_file', 'logits_file']:
        build_dir(config['params'][key])
    if config['model'] == 'resizer':
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
    eval(config)
def build_dir(save_prefix):
    p = pathlib.Path(save_prefix).parent
    p.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    main()
