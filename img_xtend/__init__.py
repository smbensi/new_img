import os
import json 

from img_xtend.utils import is_docker, ROOT_PARENT, LOGGER


version_file = f"{ROOT_PARENT}/version.json"
 
try:   
    with open(version_file, 'r') as f:
        file = json.load(f)
except json.decoder.JSONDecodeError as e:
    LOGGER.debug(f"ERROR loading version.json")
    file = {}
    
if "major" not in file.keys():
    file = {"major": 1, "minor": 0, "build": 18, "creation": "11 December 2022 11:48"}
    
__version__=f"{file['major']}.{file['minor']}.{file['build']}"