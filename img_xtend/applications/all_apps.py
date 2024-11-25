
import numpy as np

from img_xtend.utils import LOGGER, ROOT_PARENT
from img_xtend.data.build import load_inference_source
from img_xtend.detection.yolo_predictor import YoloV8
from img_xtend.tracker.reid_model import ReIDModel

from img_xtend.pipelines.face_recognition.recognize_face import FaceRecognition


# Load or check that the models are up

# Load data from the DB or config files if needed

# Connect to MQTT if needed and send alive


RUN_TRACKING=True
RUN_FACE_RECOGNITION=True
RUN_DETECTION=True
# Create different object that will load the different models
if RUN_DETECTION:
    object_detection_model = YoloV8(pose=False)
    from img_xtend.pipelines.object_detection.run_object_detection import ObjectDetectorClient
    object_detector_client = ObjectDetectorClient(object_detection_model)

if RUN_FACE_RECOGNITION or RUN_TRACKING:
    pose_estimation_model = YoloV8(pose=True)
    if RUN_TRACKING:
        reid_model = ReIDModel()
        from img_xtend.pipelines.follow_person.run_follow import FollowTracker
        tracker = FollowTracker(reid_model)
    if RUN_FACE_RECOGNITION:
        recognition = FaceRecognition()    

source = "0"
# source = f"{ROOT_PARENT}/shared/Jakes_photos/20230724_112512.jpg"
# source = f"{ROOT_PARENT}/scripts/img_mat.jpg"
# source = f"{ROOT_PARENT}/scripts/face_jake.jpg"

dataset = load_inference_source(source)  # load the source of images
LOGGER.debug(f"{source=} and {dataset.source_type=}")


for element in dataset: # return list(self.sources), list(images), [""] * self.bs
    
    img = element[1][0]
    if not isinstance(img, np.ndarray):
        LOGGER.warning("The img is not a np array")
        continue
    
    # Run inferences on the models
    if RUN_TRACKING or RUN_FACE_RECOGNITION:
        pose_results = pose_estimation_model.predict(img)
        
        if RUN_TRACKING:
            tracking_results = tracker.update(img, pose_results)
        if RUN_FACE_RECOGNITION:
            recognition_results = recognition.update(img, pose_results)
    if RUN_DETECTION:
        object_detector_client.update(img)
        
    # post-process the results
