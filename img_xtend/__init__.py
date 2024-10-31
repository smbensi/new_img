import os
import json 

from img_xtend.utils import is_docker, ROOT_PARENT

if is_docker():
    version_file = "/code/version.json"
else:
    version_file = f"{ROOT_PARENT}/version.json"
    
with open(version_file, 'r') as f:
    file = json.load(f)
    
__version__=f"{file['major']}.{file['minor']}.{file['build']}"