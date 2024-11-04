import os

### ************************
# MQTT PARAMETERS  
#### ***********************
FACE_CLIENT = None


BROKER = "localhost"
PORT = 1883

TOPICS_FROM_BRAIN = {
    "IS_ALIVE":"/robot/from_brain/are_you_alive",
    "DATA_UPDATED":"/robot/from_brain/data_updated",
    "BUILD_FACE_VECTOR":"/robot/from_brain/img/build_face_vector",
    "FACE_TIMEOUT":"/robot/from_brain/img/setfacetimeout",
    "NEW_SEARCH":"/robot/from_brain/img/search_for_face",
    "CAM_ACTIVE":"/robot/from_brain/general/camera_active",
    "STOP_SERVICE":"/robot/from_brain/img/stop_searching" 
}

TOPICS_TO_BRAIN = {
    "FACE_RECOGNITION": "/robot/to_brain/img/facefound",
    "APP_ALIVE":"/robot/to_brain/alive",
    "FACE_VECTOR_BUILT":"/robot/to_brain/face_vector_built"
}
msg_alive = {"app":"img","alive":True}


### ************************
# APPLICATIONS ACTIVATION  
#### ***********************
DISABLED = False
DO_RECOGNITION = True
ADD_NEW = False

### ************************
# PARAMETERS FOR  VIDEO STREAMING 
#### ***********************
# Wait time before trying to capture the video stream again
RETRY_STREAM = 10 if not "RETRY_STREAM" in os.environ else float(os.environ["RETRY_STREAM"])
INFO = None

#### ************************
# DETECTION PARAMETERS
#### *************************

OVERLAY = None


#### ************************
# RECOGNITION PARAMETERS
#### *************************
# ENGINE_FILE_RECOGNITION = './img_xtend/models/resnet_19_pers_fp16.trt'
ENGINE_FILE_RECOGNITION = './img_xtend/models/resnet-fp16-jetpack5_1.trt'
if os.environ["JETSON_MODEL"] == "orin":
    ENGINE_FILE_RECOGNITION = './img_xtend/models/resnet_orin_5_1.trt'

# threshold telling if there is a similiraty between 2 faces HYPERPARAMETER 
# (to be fine-tuned)
SIMILARITY_THRESHOLD = 0.75
NEW_UNKNOWN_PERSON = 1.2
SIMILARITY_THRESHOLD_UNKNOWN = 0.9

MULT_TIMEOUT = 5

DEFAULT_NULL_TIMEOUT = 10
# DEFAULT_NULL_TIMEOUT = 10_000
NULL_TIMEOUT = DEFAULT_NULL_TIMEOUT

DEFAULT_FRAMES_BEFORE_RECOGNITION = 4
FRAMES_BEFORE_RECOGNITION = DEFAULT_FRAMES_BEFORE_RECOGNITION

DEFAULT_MARGIN_IN_FRAME_TO_OMIT = 1/3
# DEFAULT_MARGIN_IN_FRAME_TO_OMIT = 0
MARGIN_IN_FRAME_TO_OMIT = DEFAULT_MARGIN_IN_FRAME_TO_OMIT
# FROM VERSION 2 for new version

RECO_FACE = None
ADD_FACE = None

DEFAULT_SEND_REC = False
# DEFAULT_SEND_REC = True
SEND_REC = DEFAULT_SEND_REC

DEFAULT_MSG_SPEED = -1
# DEFAULT_MSG_SPEED = 2
MSG_SPEED = DEFAULT_MSG_SPEED

ALL_FACE_VECTORS = None
COLLECTIONS_TO_RECOGNIZE = []

# to be deleted ?
FACE_OBJ=None # used in old version


#### ************************
# ADD NEW PERSON PARAMETERS
#### *************************
RTSP_STREAM="/dev/video0" if not "RTSP_STREAM" in os.environ else os.environ["RTSP_STREAM"]
THRESHOLD = 0.15
NETWORK = "resnet18-body"
OVERLAY = "none"
IMAGES_PATH = None
VECTORS_PATH = None
ID = None
COLLECTION = None
# PICTURES_PER_PERSON = 10 # it must not depend on this param since 7/12

#### ************************
# ADD NEW PERSON PARAMETERS
#### *************************
# DATA_SOURCE = "DB" if not "DATA_SOURCE" in os.environ else os.environ["DATA_SOURCE"]
DATA_SOURCE = os.environ.get('DATA_SOURCE', "DB") # default value is DB
 
#### ************************
# UNKNOWN GREETING PARAMETERS
#### *************************
GREETING_TIMEOUT = 5
GREETING_TRIG = 3
VALID_FACED_WIDTH = 200 # in pixels
UNKNOWN_FRAMES=20


### ************************
# FOR DEBUGGING ONLY
#### *************************
PRINT_DEBUG = (os.getenv('PRINT_DEBUG', 'False') == 'True')