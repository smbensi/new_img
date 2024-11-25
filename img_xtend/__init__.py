import os
import json 

from img_xtend.utils import is_docker, ROOT_PARENT


version_file = f"{ROOT_PARENT}/version.json"
    
with open(version_file, 'r') as f:
    file = json.load(f)
    
__version__=f"{file['major']}.{file['minor']}.{file['build']}"