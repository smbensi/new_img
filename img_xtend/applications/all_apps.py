
import numpy as np

from img_xtend.utils import LOGGER, ROOT_PARENT
from img_xtend.data.build import load_inference_source
from img_xtend.detection.predictor import YoloV8
from img_xtend.pipelines.face_recognition.recognize_face import FaceRecognition
from img_xtend.tracker.follow_tracker import FollowTracker

# Load or check that the models are up

# Load data from the DB or config files if needed

# Connect to MQTT if needed and send alive


# Create different object that will load the different models
# object_detection = YoloV8(pose=False)
pose_estimation = YoloV8(pose=True)
recognition = FaceRecognition()
# tracker = FollowTracker()

source = "0"
# source = f"{ROOT_PARENT}/shared/Jakes_photos/20230724_112512.jpg"
# source = f"{ROOT_PARENT}/scripts/img_mat.jpg"
# source = f"{ROOT_PARENT}/scripts/face_jake.jpg"
dataset = load_inference_source(source)  # load the source of images

RUN_TRACKING=False
RUN_FACE_RECOGNITION=True
RUN_DETECTION=False

for element in dataset:
    # print(element,"\n")
    img = element[1][0]
    if not isinstance(img, np.ndarray):
        continue
    if RUN_TRACKING or RUN_FACE_RECOGNITION:
        pose_results = pose_estimation.predict(img)
        # if RUN_TRACKING:
        #     tracking_results = tracker.update(pose_results)
        if RUN_FACE_RECOGNITION:
            recognition_results = recognition.update(img, pose_results)
    # if RUN_DETECTION:
    #     objects_results = object_detection.update(element)
