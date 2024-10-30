#!/usr/bin/env python3
import os
import yaml
from collections import namedtuple

from img_xtend.utils import ROOT

def get_config(tracker_type, config_file):
    assert os.path.isfile(config_file)
 
    config = {}
    with open(config_file, "r") as fo:
        config = yaml.load(fo.read(), Loader=yaml.FullLoader)

    return to_obj(tracker_type, config.get(tracker_type, {}))

def to_obj(tracker_type, dict):
    namedtuple_config = namedtuple(tracker_type, dict.keys())
    return namedtuple_config(**dict)


weights_path = f'{ROOT}/tracker/weights'
config_path = f'{ROOT}/tracker/config/config.yaml'