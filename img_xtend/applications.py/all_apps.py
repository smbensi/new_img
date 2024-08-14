
from img_xtend.utils import LOGGER
from img_xtend.data.build import load_inference_source



# Load or check that the models are up

# Load data from the DB or config files if needed

# Connect to MQTT if needed and send alive

source = "0"
dataset = load_inference_source(source)


for element in dataset:
    if RUN_TRACKING:
        pass
    if RUN_FACE_RECOGNITION:
        pass
    if RUN_DETECTION:
        pass
