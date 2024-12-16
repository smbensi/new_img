import json 

from img_xtend.utils import is_docker, LOGGER

CONFIDENCE = 0.5
CLASS_FILTER = None
DEBUG_LAST_CLS_PUBLISHED = []
MSG_SEND_INTERVAL = 2 # seconds 

# TODO a remplacer
if is_docker():
    path = '/code/shared/ml-artifacts/yolov8/objects.json'
    # path = '/code/shared/ml-artifacts/object_detection/objects.json'
else:
    path = '/home/nvidia/xtend/shared/ml-artifacts/yolov8/objects.json'
with open(path) as f:
    data = json.load(f)
list_of_objects = data["objects"]
ids = [el["_id"] for el in list_of_objects]
names = [el["name"] for el in list_of_objects]

LABELS = dict(zip(ids, names))
LOGGER.debug(f"OBJECT {LABELS=}")
inverted_labels = dict(zip(names,ids))