import importlib
from os import confstr

def parse(config_file_name):
    opt = importlib.import_module("conf."+config_file_name).get()
    return opt
