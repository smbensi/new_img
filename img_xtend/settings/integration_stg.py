import os
import yaml

# Load data from the DB or config files if needed
config = {}
config_file = "img_xtend/applications/all_apps_cfg.yaml"
assert os.path.isfile(config_file)
with open(config_file, "r") as fo:
    config = yaml.load(fo.read(), Loader=yaml.FullLoader)

USE_TRITON=True
RUN_TRACKING=config["INTEGRATION"]["RUN_FOLLOW_ME"]
RUN_DETECTION=config["INTEGRATION"]["RUN_OBJECT_DETECTION"]
RUN_FACE_RECOGNITION=config["INTEGRATION"]["RUN_FACE_RECOGNITION"]