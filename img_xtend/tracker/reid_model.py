import os
import gdown
from pathlib import Path

import numpy as np
import torch

from img_xtend.utils import LOGGER, triton
from img_xtend.tracker.config.config_utils import weights_path, config_path, get_config
from img_xtend.tracker.appearance.reid_auto_backend import ReidAutoBackend

def download_weights(type):
    if not os.path.isdir(weights_path):
        os.mkdir(weights_path)

    if type == 'OSNET':
        # https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
        reid_weights = weights_path + '/osnet_x1_0.pt'
        url = 'https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY'
    #if type == 'OSNET':
    #    reid_weights = weights_path + '/osnet_x0_25_msmt17.pt'
    #    url = 'https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF'
    elif type == 'MOBILENET':
        reid_weights = weights_path + '/mobilenetv2_x1_4_dukemtmcreid.pt'
        url = 'https://drive.google.com/uc?id=1t6JCqphJG-fwwPVkRLmGGyEBhGOf2GO5'
    elif type == 'RESNET':
        reid_weights = weights_path + '/resnet50_msmt17.pt'
        url = 'https://drive.google.com/uc?id=1yiBteqgIZoOeywE8AhGmEQl7FTVwrQmf'
    else:
        reid_weights = ''
        url = ''

    reid_weights = Path(reid_weights).resolve()
    if not reid_weights.exists() and reid_weights != '' and url != '':
        output = str(reid_weights)
        gdown.download(url, output, quiet=False)

    return reid_weights


class ReIDModel():
    def __init__(self) -> None:
        
        cfg = get_config(tracker_type, config_path)
        
        if USE_TRITON:
            self.half = False
            path = "http://localhost:8000/osnet_x1_0"
            self.model = triton.TritonRemoteModel(path)
            self.device = 'cpu'
        else:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            tracker_type = tracker_settings.tracker_name
            reid_weights = download_weights(cfg.reid_weights)

            rab = ReidAutoBackend(
            weights=reid_weights, 
            device=self.device,
            half=self.fp16
            )
        
            self.model = rab.get_backend()   
            self.warmup() 
    
    def get_features(self, crops: np.ndarray):
        
        if crops.shape[0] != 0:
            features = self.model.forward(crops)
        else:
            features = np.array([])
        features = features / np.linalg.norm(features)
        return features
        
    def warmup(self, imgsz=[(256, 128, 3)]):
        # warmup model by running inference once
        # if self.device.type != "cpu":
        im = np.random.randint(0, 255, *imgsz, dtype=np.uint8)
        im = self.get_crops(xyxys=np.array(
            [[0, 0, 64, 64], [0, 0, 128, 128]]),
            img=im
        )
        self.forward(im)  # warmup