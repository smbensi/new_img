"""all tasks linked to object detection"""

import os
import time

import numpy as np
import cv2
from ultralytics import YOLO
import tritonclient

from img_xtend.utils import LOGGER, is_docker, ROOT_PARENT
from img_xtend.settings.integration_stg import *
from img_xtend.detection.bbox import Bbox

class YoloV8:
    '''class responsible of load and predict all YoloV8 models (object detection and pose estimation)'''
    
    def __init__(self, pose:bool=False):
        start = time.time()
        model_folder = "/code/models" if is_docker() else f'{ROOT_PARENT}/models/pose_estimation' # FIXME verifier que c'est la bonne adresse
        try:
            if pose:
                if USE_TRITON:
                    LOGGER.debug("****POSE MODEL FROM TRITON****")
                    model_path = f'http://localhost:8000/yoloV8_pose'
                else:
                    model_name = "yolov8n-pose.pt"
                    model_path = f'{model_folder}/{model_name}'

                self.model = YOLO(model_path,task='pose')
                    
                var_names = {0: 'person'}
                self.model.predictor = self.model._smart_load("predictor")()
                self.model.predictor.setup_model(model=self.model.model) # TODO cette ligne imprime Ultralytics YOLOv8.0.222 🚀 Python-3.8.16 torch-2.2.2 CPU (ARMv8 Processor rev 0 (v8l))
                self.model.predictor.model.names = var_names
                self.model.predictor.model.kpt_shape = [17, 3]

                

            else:
                if USE_TRITON:
                    LOGGER.debug("****DETECTION MODEL FROM TRITON****")
                    model_path = f'http://localhost:8000/yolov8'
                else:
                    LOGGER.debug("****DETECTION MODEL IS LOCAL (NOT TRITON)****")
                    model_name = os.getenv("MODEL_NAME","yolov8n.pt") #FP16.engine")
                    model_path = f'{model_folder}/{model_name}'

                self.model = YOLO(model_path,task='detect')

            self.warmup()
        except ConnectionRefusedError as e:
            print(f"ERROR: {e} \nCheck that the Triton server is correctly loaded")
            exit()
        except tritonclient.utils.InferenceServerException as e:
            print(f"ERROR: {e} \nCheck that the model is correctly loaded in the Triton Server")
            exit()
        LOGGER.debug(f"WARMUP TIME:{model_path= } {time.time() - start:.1f} sec")
    
    def warmup(self):
        """Run a fake example to make the model ready to accept new input"""
        img = cv2.imread("img_xtend/detection/man.jpg") 
        bboxes = self.predict(img)
        LOGGER.debug(bboxes)
    
    def predict(self,img:np.ndarray,conf:float=0.5,class_filter:list=None,show:bool=False,stream:bool=False,verbose:bool=False, track:bool=False):
        if track:
            results = self.model.track(source=img, conf=conf, classes=class_filter, save=False, stream=stream, verbose=verbose, show=show)

        else:
            results = self.model.predict(source=img, conf=conf, classes=class_filter, save=False, stream=False, verbose=verbose, show=show)
        
        # logger.debug(f"{results=}")
        self.height, self.width = img.shape[:2]
        bboxes = self.result_to_bbox(results)
        
        return bboxes

    def result_to_bbox(self, results):
        '''Convert the results returned by YoloV8 into Bbox dataclass'''
        
        bboxes = []
        id = 2000
        
        keypoints_result = results[0].keypoints if  hasattr(results[0],'keypoints') else None
        if keypoints_result is None:
            keypoints_result = [[] for _ in range(len(results[0].boxes))]
            
        for box, keypoint in zip(results[0].boxes, keypoints_result):
            try:
                cls = int(box.cls)
            except:
                raise TypeError("The class needs to be an integer")
            
            conf = float(box.conf)
            x, y, w, h = box.xywh.int().tolist()[0]
            
            id_test = box.id
            if id_test != None:
                id = id_test.int().item()
            else:
                id += 1
                
            bbox = Bbox(x,y,w,h,id,conf,cls, height_frame=self.height, width_frame=self.width, keypoints=keypoint)
            bboxes.append(bbox)
            
        return bboxes