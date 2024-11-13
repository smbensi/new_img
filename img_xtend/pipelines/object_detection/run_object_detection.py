import json
import time

import numpy as np

from img_xtend.utils import LOGGER
from img_xtend.mqtt import mqtt_handler, mqtt_settings
from img_xtend.detection.yolo_predictor import YoloV8
from img_xtend.detection.bbox import Bbox
from img_xtend.pipelines.object_detection import object_detection_settings


def mqtt_msg_for_rock(bboxes:Bbox,track=False,for_debug=False):
    mqtt_msg = []
    debug_cls = []
    for i,bbox in enumerate(bboxes):
        cls = bbox.cls
        cls_text = object_detection_settings.LABELS[cls]
        id = bbox.id
        if for_debug:
            id +=1
        debug_cls.append(cls)
        posn = {"x":bbox.x, "y":bbox.y, "w":bbox.w, "h":bbox.h}
        bbox_data = {"class":cls_text, "posn":posn, "id":id}
        mqtt_msg.append(bbox_data)
    
    if for_debug:
        return mqtt_msg
    msg = {
        "obj_found":mqtt_msg
    }
    # if settings.DEBUG_LAST_CLS_PUBLISHED != debug_cls:
    #     LOGGER.debug(msg)
    # settings.DEBUG_LAST_CLS_PUBLISHED = debug_cls
    
    return msg

class ObjectDetectorClient:
    def __init__(self, model):
        self.model = model
        self.mqtt_client = mqtt_handler.init_mqtt_connection('object_detection_client')
        self.last_publish = time.time()
        
    def update(self, img):
        # predict
        if not isinstance(img, np.ndarray):
            LOGGER.warning(f"the input to the object detection model is a {type(img)} and is not a nparray")
            return()
        
        track = True
        results = self.model.predict(img, class_filter=object_detection_settings.CLASS_FILTER, track=track)
        mqtt_msg = mqtt_msg_for_rock(results)
        
        if time.time() - self.last_publish > object_detection_settings.MSG_SEND_INTERVAL:
            LOGGER.debug(f"msg to publish {mqtt_msg}")
            self.last_publish = time.time()
            self.mqtt_client.publish(mqtt_settings.TOPICS_TO_BRAIN['DETECTION_FDBK'], json.dumps(mqtt_msg))