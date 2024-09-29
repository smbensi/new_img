
from img_xtend.utils import LOGGER
from img_xtend.data.build import load_inference_source
from img_xtend.detection.predictor import YoloV8


# Load or check that the models are up

# Load data from the DB or config files if needed

# Connect to MQTT if needed and send alive

source = "0"
dataset = load_inference_source(source)

object_detection_model = YoloV8(pose=False)
tracker_model = YoloV8(pose=True)

RUN_TRACKING=False
RUN_FACE_RECOGNITION=False
RUN_DETECTION=False

for element in dataset:
    # print(element,"\n")
    if RUN_TRACKING:
        pass
    if RUN_FACE_RECOGNITION:
        pass
    if RUN_DETECTION:
        pass
