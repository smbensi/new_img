""" All the MQTT parameters"""

PORT = 1883
BROKER = "localhost"
OBJ_CLIENT = None

# msg template for ACTIVATE_DETECTION the filter corresponds to the classes
# we are looking for
# {
#    “obj_detection” : “on”/ “off’,
#    “filter” : [ “class1”, “class2”..]
# }

TOPICS_FROM_BRAIN = {
    "ACTIVATE_DETECTION":"/robot/from_brain/img/obj_detection",
    "IS_ALIVE":"/robot/from_brain/are_you_alive",
    "ACTIVATE_FOLLOW_ME":"robot/ros2/set/follow_person_mode",
}

# msg template for DETECTION_FDBK
# {
#       “obj_found” : [
#           “obj” : {
#                 “class” : <name>,
#                 “posn” : { “x”,”y”,”w”,”h” }
#           },
#           …
#       ]
# }

TOPICS_TO_BRAIN = {
    "DETECTION_FDBK": "/robot/to_brain/img/obj_found",
    "APP_ALIVE":"/robot/to_brain/alive",
}

TOPICS_TO_ROS = {
    "BBOX_OBJECT":"robot/ros2/objectFound",
}
msg_alive = {"app":"object_detection","alive":True}
