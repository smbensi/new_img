import json
import threading
import time
import logging

import paho.mqtt.client as mqtt

from img_xtend.utils import LOGGER

import img_xtend.mqtt.mqtt_settings as params
# from img_xtend.pipelines.object_detection import object_detection_settings

# logger = logging.getLogger(__name__)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.connected_flag = True
        client.cleanSession = True
        LOGGER.debug(f"*** Connected successfully to mqtt and {topics_to_subscribe}")
        # topics_to_sub = [(topic,2) for topic in params.TOPICS_FROM_BRAIN.values()]
        # topics_to_sub += [(topic,2) for topic in params.TOPICS_TO_BRAIN.values()]
        # topics_to_sub += [(topic,2) for topic in params.TOPICS_TO_ROS.values()]
        client.subscribe(topics_to_subscribe)
    else:
        LOGGER.debug("*** Couldnt connect to mqtt, Error code: " + rc)
        client.loop_stop()

def on_disconnect(client, userdata, rc):
    LOGGER.debug("Disconnected from mqtt")
    # client.loop_stop()

def connect_to_broker(userdata):
    # Keep Alive params: The keep-alive interval, which is the maximum time interval between communications with the broker.
    # If the client does not communicate within this time frame, the broker may consider the client disconnected.
    client.connect(userdata["INTEGRATION"]["BROKER"], 
                   userdata["INTEGRATION"]["PORT"],
                   keepalive=userdata["INTEGRATION"]["keepalive"])
    client.loop_start()

def disconnect():
    client.disconnect()
    client.loop_stop()

def init_mqtt_connection(name, topics_to_sub, cfg, on_message):
    mqtt.Client.connected_flag = False
    global client
    global topics_to_subscribe
    topics_to_subscribe = topics_to_sub
    client = mqtt.Client(name, clean_session=True)
    client.user_data_set(cfg)
    client.on_disconnect = on_disconnect
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect_to_broker = connect_to_broker
    client.connect_to_broker(client._userdata)
    return client

# def on_message(client, userdata, message):

#     decoded_message = (message.payload.decode("utf-8"))
    
#     if message.topic == params.TOPICS_FROM_BRAIN["ACTIVATE_DETECTION"]:
#         decoded_message = json.loads(decoded_message)
#         object_detection_settings.MODE = "DETECTION"
#         LOGGER.debug(f"msg for APP_ALIVE topic : {decoded_message}")
#         if decoded_message["obj_detection"] == "on":
#             object_detection_settings.RUN_DETECTION = True
#             class_filter = decoded_message.get("filter",[])
#             print(f"{class_filter=}")
#             isints = [0]
#             isnums = [0]
            
#             if class_filter and isinstance(class_filter[0],int):
#                 isints = [isinstance(el,int) for el in class_filter]
#             elif class_filter and isinstance(class_filter[0],str):
#                 isnums = [el.isnumeric() for el in class_filter]
#                 # print(isnum)
#             if class_filter and (all(isnums) or all(isints)):
#                 object_detection_settings.CLASS_FILTER = [int(el) for el in class_filter if int(el)<len(object_detection_settings.inverted_labels)]
#             else:
#                 object_detection_settings.CLASS_FILTER = [object_detection_settings.inverted_labels[el.strip()]
#                                                     for el in class_filter
#                                                     if el.strip() in object_detection_settings.inverted_labels]
#             print(f'{object_detection_settings.CLASS_FILTER=}')
#         elif decoded_message["obj_detection"] == "off":
#             object_detection_settings.RUN_DETECTION = False
            
#     elif message.topic == params.TOPICS_TO_BRAIN["DETECTION_FDBK"]:
#         # print(decoded_message)
#         pass
    
#     elif message.topic == params.TOPICS_TO_ROS["BBOX_OBJECT"]:
#         # LOGGER.debug(f"TO ROS: {decoded_message}")
#         pass
        
#     elif message.topic == params.TOPICS_FROM_BRAIN["IS_ALIVE"]:
#         params.OBJ_CLIENT.publish(params.TOPICS_TO_BRAIN["APP_ALIVE"],json.dumps(params.msg_alive))

#     elif message.topic == params.TOPICS_TO_BRAIN["APP_ALIVE"]:
#         print(json.loads(decoded_message))
    
#     elif message.topic == params.TOPICS_FROM_BRAIN["ACTIVATE_FOLLOW_ME"]:
#         LOGGER.debug(f"msg for FOLLOW topic : {decoded_message} and type= {type(decoded_message)}")
#         if isinstance(decoded_message, str):
#             LOGGER.debug("the message is a string")
            
#         if decoded_message == "True":
#             object_detection_settings.MODE = "FOLLOW"
#             object_detection_settings.RUN_DETECTION = True
#             object_detection_settings.INIT_TRACKER = True
#             object_detection_settings.MATCHES_SOURCE = {"FRAMES":0, "IOU":0, "APPEARANCE":0,"MANUAL":0}
#         elif decoded_message == "False":
#             object_detection_settings.MODE = "DETECTION"
#             object_detection_settings.RUN_DETECTION = False
#             appearance_pourcentage = 100* object_detection_settings.MATCHES_SOURCE["APPEARANCE"] / object_detection_settings.MATCHES_SOURCE["FRAMES"]
#             manual_pourcentage = 100* object_detection_settings.MATCHES_SOURCE["MANUAL"] / object_detection_settings.MATCHES_SOURCE["FRAMES"]
#             iou_pourcentage = 100* object_detection_settings.MATCHES_SOURCE["IOU"] / object_detection_settings.MATCHES_SOURCE["FRAMES"] 
#             LOGGER.debug(f"STATS: {object_detection_settings.MATCHES_SOURCE} \n Appearance={appearance_pourcentage:.0f}%  IOU={iou_pourcentage:.0f}% MANUAL={manual_pourcentage:.0f}%")