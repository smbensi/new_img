from pathlib import Path
import os

tracker_name = 'STRONG_SORT'
bbox_output = 'numpy'

FILE = Path(__file__).resolve()
PACKAGE_ROOT = FILE.parents[1]
WEIGHTS = PACKAGE_ROOT / "tracker"  / "weights"

DEFAULT_CONFIG_FILE = PACKAGE_ROOT / "tracker/config/config.yaml"
CONFIG_FILE = os.getenv("CONFIG_FILE",DEFAULT_CONFIG_FILE)
USE_MANUAL_MATCH = (os.getenv('USE_MANUAL_MATCH', 'True') == 'True')

NB_OF_FEATURES_VECTORS = 30
NB_OF_MISS_IOU = 5

