# Create all the config files for a run without using smt
import sys
import os
import yaml
import uuid
import hashlib

def get_config_label(config, characters=8, ignore_label_key=True):
    # Returns a unique label for a specific dictionary of parameters
    # no. of characters can be specified optionally
    # The dictionary entry 'label' is ignored here
    config_sorted = dict()
    # Sort dict first
    for key in sorted(config):
        if key != 'label' and ignore_label_key:
            config_sorted[key] = config[key]
    hashstring = hashlib.sha1(repr(config_sorted).encode()).hexdigest()
    return hashstring[:characters]

def create_config_dict(**kwargs):
    default = yaml.load(open('default_config.yaml'))
    for key in kwargs:
        value = kwargs[key]
        if key in default:
            try:
                value = int(value)
            except:
                try:
                    value = float(value)
                except:
                    value = value
                    if value == 'False':
                        value = False
                    if value == 'True':
                        value = True
            default[key] = value
    default['label'] = get_config_label(default)
    return default

def argv2dict(argv):
    adict = dict()
    for arg in argv:
        [key,value] = arg.split('=')
        adict[key] = value
    return adict

def create_config_file(argv):
    "Creates a config file from command line arguments and returns the label"
    avdict = argv2dict(argv[1:])
    config = create_config_dict(**avdict)
    label = config['label']
    subdir = 'config_files/'

    if not os.path.isdir(subdir):
        os.makedirs(subdir)

    yaml.dump(config, open(subdir + '{}.yaml'.format(label), 'w'))
    return label
