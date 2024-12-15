import os
import numpy as np

from img_xtend.utils import LOGGER
from img_xtend.settings import integration_stg

from img_xtend.data.build import load_inference_source  # load the source of images

from img_xtend.detection.yolo_predictor import YoloV8
from img_xtend.tracker.reid_model import ReIDModel # 6 sec
from img_xtend.pipelines.face_recognition.run_face_recognition import FaceRecognitionClient


# Load data from the DB or config files if needed
config = integration_stg.config


# check if we need to load models
LOAD_TRACKING = config["INTEGRATION"]["LOAD_MODELS_FOLLOW_ME"]
LOAD_DETECTION = config["INTEGRATION"]["LOAD_MODELS_OBJECT_DETECTION"]
LOAD_FACE_RECOGNITION = config["INTEGRATION"]["LOAD_MODELS_FACE_RECOGNITON"]

# Create different object that will load the different models
if LOAD_DETECTION:
    object_detection_model = YoloV8(pose=False)
    from img_xtend.pipelines.object_detection.run_object_detection import ObjectDetectorClient
    object_detector_client = ObjectDetectorClient(object_detection_model,config)

if LOAD_FACE_RECOGNITION or LOAD_TRACKING:
    pose_estimation_model = YoloV8(pose=True)
    if LOAD_TRACKING:
        reid_model = ReIDModel()
        from img_xtend.pipelines.follow_person.run_follow import FollowTracker
        tracker = FollowTracker(reid_model,config)
    if LOAD_FACE_RECOGNITION:
        recognition = FaceRecognitionClient()    

source = os.getenv("SOURCE",config["INTEGRATION"]["source"])
# source = f"{ROOT_PARENT}/shared/Jakes_photos/20230724_112512.jpg"
# source = f"{ROOT_PARENT}/scripts/img_mat.jpg"
# source = f"{ROOT_PARENT}/scripts/face_jake.jpg"
LOGGER.debug(f"video source is {source}")

dataset = load_inference_source(source)  # load the source of images
LOGGER.debug(f"{source=} and {dataset.source_type=}")

for element in dataset: # return list(self.sources), list(images), [""] * self.bs
    
    img = element[1][0] # get only the image of the first source
    if not isinstance(img, np.ndarray):
        LOGGER.warning("The img is not a np array")
        continue
    
    # Run inferences on the models
    if integration_stg.RUN_TRACKING or integration_stg.RUN_FACE_RECOGNITION:
        pose_results = pose_estimation_model.predict(img)
        
        if integration_stg.RUN_TRACKING:
            tracking_results = tracker.update(img, pose_results)
        if integration_stg.RUN_FACE_RECOGNITION:
            recognition_results = recognition.update(img, pose_results)
    if integration_stg.RUN_DETECTION:
        object_detector_client.update(img)
        
    # post-process the results
