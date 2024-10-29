import os
import json
from pathlib import Path

from img_xtend.utils import get_time, is_docker

FILE = Path(__file__)
ROOT = FILE.parents[2]
CLASS_FILTER = ''
CONFIDENCE=float(os.getenv("CONFIDENCE","0.5"))
TIME_INTERVAL = float(os.getenv("TIME_INTERVAL","1"))
SOURCE = os.getenv("SOURCE","/dev/video0")
RUN_DETECTION = (os.getenv('RUN_DETECTION', 'False') == 'True')
MODE = os.getenv("MODE","DETECTION")
INIT_TRACKER = False
TRACKER = os.getenv('TRACKER', 'STRONG_SORT')
SHOW_YOLO_DETECTION = (os.getenv('SHOW_YOLO_DETECTION', 'True') == 'True')

if is_docker():
    path = '/code/shared/ml-artifacts/object_detection/objects.json'
else:
    path = '/home/nvidia/dev/nlp/tests/test_data/objects.json'
    
with open(path) as f:
    data = json.load(f)
list_of_objects = data["objects"]
ids = [el["_id"] for el in list_of_objects]
names = [el["name"] for el in list_of_objects]

ULTRALYTICS_LABELS = dict(zip(ids, names))
inverted_labels = dict(zip(names,ids))

# visualization
SHOW_RESULTS = int(os.getenv("SHOW_RESULTS", "0"))

# DEBUGGING SETTINGS
MATCHES_SOURCE = {"FRAMES":0, "IOU":0, "APPEARANCE":0,"MANUAL":0}
DEBUG_LAST_CLS_PUBLISHED = []
MATCHES_TO_PRINT = os.getenv("MATCHES_TO_PRINT",'APPERANCE,IOU,MANUAL').replace(" ","").split(',')
DETECTIONS_FOLDER_PATH = ROOT / "shared/for_debugging"
# DETECTIONS_FOLDER_PATH = "/code/shared/for_debugging"
os.makedirs(DETECTIONS_FOLDER_PATH,exist_ok=True)

# file containing all the text info when a new detection added
NEW_DETECTION_INFO_FILE = "/code/shared/for_debugging/new_detection_info.txt"
NEW_DETECTION_INFO_FILE = ROOT / "shared/for_debugging/new_detection_info.txt"
with open(NEW_DETECTION_INFO_FILE, "w") as file:
    file.write(f"{get_time()} NEW RUN \n")
    
IOU_LOGS = ROOT / "shared/for_debugging/iou_logs.txt"
with open(IOU_LOGS, "w") as file:
    file.write(f"{get_time()} NEW IOU_LOGS \n")




# print(f'{ULTRALYTICS_LABELS=}')
# Ultralytics basic model labels
# ULTRALYTICS_LABELS = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
#  4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
#  8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
#  11: 'stop sign', 12: 'parking meter', 13: 'bench',
#  14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
#  18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
#  22: 'zebra', 23: 'giraffe', 24: 'backpack', 
#  25: 'umbrella', 26: 'handbag', 27: 'tie',
#  28: 'suitcase', 29: 'frisbee', 30: 'skis',
#  31: 'snowboard', 32: 'sports ball',
#  33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
#  36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
#  39: 'bottle', 40: 'wine glass', 41: 'cup', 
#  42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 
#  46: 'banana', 47: 'apple', 48: 'sandwich', 
#  49: 'orange', 50: 'broccoli', 51: 'carrot', 
#  52: 'hot dog', 53: 'pizza', 54: 'donut',
#  55: 'cake', 56: 'chair', 57: 'couch',
#  58: 'potted plant', 59: 'bed', 60: 'dining table',
#  61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
#  66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
#  71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 
#  76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


# inverted_labels = {value: key for key, value in ULTRALYTICS_LABELS.items()}

