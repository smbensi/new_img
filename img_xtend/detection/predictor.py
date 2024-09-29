"""all tasks linked to object detection"""

import os
import time

import numpy as np
import cv2
from ultralytics import YOLO

from img_xtend.utils import LOGGER, is_docker
from img_xtend.settings.integration_stg import *
from .bbox import Bbox

class YoloV8:
    '''class responsible of load and predict all YoloV8 models (object detection and pose estimation)'''
    
    def __init__(self, pose:bool=False):
        start = time.time()
        model_folder = "/code/models" if is_docker() else f'{os.getcwd()}/img_xtend/models/ultralytics'
        if pose:
            model_name = "yolov8n-pose.pt"
            model_path = f'{model_folder}/{model_name}'

        else:
            if USE_TRITON:
                LOGGER.debug("****DETECTION MODEL FROM TRITON****")
                model_path = f'http://localhost:8000/ultralytics'
            else:
                LOGGER.debug("****DETECTION MODEL IS LOCAL (NOT TRITON)****")
                model_name = os.getenv("MODEL_NAME","yolov8n.pt") #FP16.engine")
                model_path = f'{model_folder}/{model_name}'
                
        self.model = YOLO(model_path,task='detect')
        
        self.warmup()
        LOGGER.debug(f"WARMUP TIME:{model_path= } {time.time() - start:.1f} sec")
    
    def warmup(self):
        """Run a fake example to make the model ready to accept new input"""
        img = cv2.imread("img_xtend/detection/man.jpg") # FIXME change with a photo with a person
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