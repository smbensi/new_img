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

def on_message(client, userdata, message):

    decoded_message = (message.payload.decode("utf-8"))
    LOGGER.debug(f"{decoded_message}")
    LOGGER.debug(f"{userdata=}")
    print(f"{userdata=}")
    LOGGER.debug(f"{message.topic=}")
    
    # if message.topic == params.TOPICS_FROM_BRAIN["ACTIVATE_DETECTION"]:
    #     decoded_message = json.loads(decoded_message)
    #     object_detection_settings.MODE = "DETECTION"
    #     LOGGER.debug(f"msg for APP_ALIVE topic : {decoded_message}")
    #     if decoded_message["obj_detection"] == "on":
    #         object_detection_settings.RUN_DETECTION = True
    #         class_filter = decoded_message.get("filter",[])
    #         print(f"{class_filter=}")
    #         isints = [0]
    #         isnums = [0]
            
    #         if class_filter and isinstance(class_filter[0],int):
    #             isints = [isinstance(el,int) for el in class_filter]
    #         elif class_filter and isinstance(class_filter[0],str):
    #             isnums = [el.isnumeric() for el in class_filter]
    #             # print(isnum)
    #         if class_filter and (all(isnums) or all(isints)):
    #             object_detection_settings.CLASS_FILTER = [int(el) for el in class_filter if int(el)<len(object_detection_settings.inverted_labels)]
    #         else:
    #             object_detection_settings.CLASS_FILTER = [object_detection_settings.inverted_labels[el.strip()]
    #                                                 for el in class_filter
    #                                                 if el.strip() in object_detection_settings.inverted_labels]
    #         print(f'{object_detection_settings.CLASS_FILTER=}')
    #     elif decoded_message["obj_detection"] == "off":
    #         object_detection_settings.RUN_DETECTION = False
            
    # elif message.topic == params.TOPICS_TO_BRAIN["DETECTION_FDBK"]:
    #     # print(decoded_message)
    #     pass
    
    # elif message.topic == params.TOPICS_TO_ROS["BBOX_OBJECT"]:
    #     # LOGGER.debug(f"TO ROS: {decoded_message}")
    #     pass
        
    # elif message.topic == config["object_detection"]["mqtt_topics"]["TOPICS_FROM_BRAIN"]["IS_ALIVE"]:
    #     params.OBJ_CLIENT.publish(params.TOPICS_TO_BRAIN["APP_ALIVE"],json.dumps(params.msg_alive))

    if message.topic == userdata["object_detection"]["TOPICS_TO_BRAIN"]["APP_ALIVE"]:
        LOGGER.debug(f"APP ALIVE {json.loads(decoded_message)}")
        print(f"APP ALIVE {json.loads(decoded_message)}")
    
    # elif message.topic == params.TOPICS_FROM_BRAIN["ACTIVATE_FOLLOW_ME"]:
    #     LOGGER.debug(f"msg for FOLLOW topic : {decoded_message} and type= {type(decoded_message)}")
    #     if isinstance(decoded_message, str):
    #         LOGGER.debug("the message is a string")
            
    #     if decoded_message == "True":
    #         object_detection_settings.MODE = "FOLLOW"
    #         object_detection_settings.RUN_DETECTION = True
    #         object_detection_settings.INIT_TRACKER = True
    #         object_detection_settings.MATCHES_SOURCE = {"FRAMES":0, "IOU":0, "APPEARANCE":0,"MANUAL":0}
    #     elif decoded_message == "False":
    #         object_detection_settings.MODE = "DETECTION"
    #         object_detection_settings.RUN_DETECTION = False
    #         appearance_pourcentage = 100* object_detection_settings.MATCHES_SOURCE["APPEARANCE"] / object_detection_settings.MATCHES_SOURCE["FRAMES"]
    #         manual_pourcentage = 100* object_detection_settings.MATCHES_SOURCE["MANUAL"] / object_detection_settings.MATCHES_SOURCE["FRAMES"]
    #         iou_pourcentage = 100* object_detection_settings.MATCHES_SOURCE["IOU"] / object_detection_settings.MATCHES_SOURCE["FRAMES"] 
    #         LOGGER.debug(f"STATS: {object_detection_settings.MATCHES_SOURCE} \n Appearance={appearance_pourcentage:.0f}%  IOU={iou_pourcentage:.0f}% MANUAL={manual_pourcentage:.0f}%")

class ObjectDetectorClient:
    def __init__(self, model, config):
        self.model = model
        if config["INTEGRATION"]["use_mqtt"]:
            topics_to_sub = [(topic,2) for topic in config["object_detection"]["TOPICS_FROM_BRAIN"].values()]
            topics_to_sub += [(topic,2) for topic in config["object_detection"]["TOPICS_TO_BRAIN"].values()]
            topics_to_sub += [(topic,2) for topic in config["object_detection"]["TOPICS_TO_ROS"].values()]
            self.mqtt_client = mqtt_handler.init_mqtt_connection('object_detection_client', topics_to_sub, config, on_message)
            self.mqtt_client.publish(config["object_detection"]["TOPICS_TO_BRAIN"]["APP_ALIVE"],json.dumps(mqtt_settings.msg_alive))
            
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