import os
from pathlib import Path
import gdown
import time
import copy

import abc
from typing import List

import numpy as np
import cv2
import torch
from img_xtend.tracker import tracker_settings

from img_xtend.utils import LOGGER, get_time
from img_xtend.tracker.config.config_utils import weights_path, config_path, get_config
from img_xtend.tracker.appearance.reid_auto_backend import ReidAutoBackend
from img_xtend.tracker.tracker import Tracker
from img_xtend.tracker.matching.match_dataclass import MatchData
from img_xtend.tracker.matching.metrics_matching import NearestNeighborDistanceMetric
from img_xtend.detection.bbox import Bbox
from img_xtend.pose_estimation import keypoints
from img_xtend.pipelines.follow_person import follow_settings
THRESHOLD_SHOULDERS = 100 # in pixels



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

def mqtt_for_rock(bboxes:Bbox,track=False,for_debug=False,id_followed=-1):
    mqtt_msg = []
    debug_cls = []
    for i,bbox in enumerate(bboxes):
        cls = bbox.cls
        cls = follow_settings.ULTRALYTICS_LABELS[cls]
        id = bbox.id
        if id == id_followed:
            continue
        if for_debug:
            id +=1
        debug_cls.append(cls)
        posn = {"x":bbox.x, "y":bbox.y, "w":bbox.w, "h":bbox.h}
        bbox_data = {"class":cls, "posn":posn, "id":id}
        mqtt_msg.append(bbox_data)
    
    if for_debug:
        return mqtt_msg
    msg = {
        "obj_found":mqtt_msg
    }
    if follow_settings.DEBUG_LAST_CLS_PUBLISHED != debug_cls:
        LOGGER.debug(msg)
    follow_settings.DEBUG_LAST_CLS_PUBLISHED = debug_cls
    return msg


def bbox_check(bbox,bboxes_yolo=None):
    """
    Checking that we have a logical bbox
    Checking the aspect ratio
    """
    # if bboxes_yolo:
    #     iou_matching = iou(bbox, bboxes_yolo).tolist()
    #     # LOGGER.debug(f"{iou_matching=}")
    #     if all(value < 0.9 for value in iou_matching):
    #         LOGGER.debug("NOT OVERLAPPING WITH A WINDOW FROM YOLO")
    #         return False
            
    aspect_ratio = bbox.w / bbox.h
    if aspect_ratio > 4 or aspect_ratio < 1/4:
        LOGGER.debug("ASPECT RATIO NOT LOGIC")
        return False
    if bbox.w < 30 or bbox.h < 30:
        LOGGER.debug("BBOX TO SMALL")
        return False
    
    area = bbox.w * bbox.h
    return True

class BaseTracker(abc.ABC):
    def __init__(self):
        
        self.width = None
        self.height = None
        self.following = False
        
        
    def choose_id(self,bboxes:List[Bbox]):
        """
        The function searchs for the bbox with the greatest area and return it plus its index

        Args:
            bboxes (List[Bbox]): List of the bboxes deteected in the frame

        Returns:
            bbox: (Bbox | None): The bbox from the list of bboxes with the greatest area
            index: (int | None): The index in the list of the bbox with the greatest area
        """
        if len(bboxes) == 0:
            return None, None
        if len(bboxes) == 1:
            return bboxes[0], bboxes[0].id
        else:
            # bboxes.sort(reverse=True, key=lambda x:x.w*x.h)
            sorted_bboxes = sorted(bboxes,reverse=True, key=lambda x:x.w*x.h)
            LOGGER.debug(f"{bboxes=}")
            LOGGER.debug(f"{sorted_bboxes=}")
            for bbox in sorted_bboxes:
                if bbox_check(bbox):
                    # for i,el in enumerate(bboxes):
                    #     if bbox.id == el.id:
                    #         index = i
                    index = bbox.id
                    return bbox, index
        return None, None

class FollowTracker(BaseTracker):
    """
    This tracker is based on the StrongSORT logic but the Kalman Filter was removed
    because we get to many errors
    
    Methods:
        - __init__ : initiates the Reid model and the Tracker
        - initialize_tracker: Re/initiates the Tracker 
        - update: the main function that controls all the pipeline
        - initiate_followed_id: if we are not following a person yet, it initiates the person to follow
        - update_followed_id: if we are following someone, it search for him in the present frame
        - set_follow_track: update the tracker to set a particular track as the followed track
        - prepare_msg_to_publish: prepare the JSON for the MQTT message
        - check_same_ids: checks that the track.id and track.bbox_info.id are the same
        - check_save_img: check the conditions to save an image for debugging
        - save_image: Save the frame with the bbox

    """
    def __init__(self) -> None:
        """
        The constructors does 2 things:
            - Loads the Re-id model
            - Initiates the Tracker
        """
        
        super().__init__()
        tracker_type = tracker_settings.tracker_name
        cfg = get_config(tracker_type, config_path)
        LOGGER.debug(f"{follow_settings.MATCHES_TO_PRINT=}")
        LOGGER.debug(f"\n")
        LOGGER.debug(f"CONFIG TRACKER: {config_path=} {cfg}")
        LOGGER.debug(f"\n")
        LOGGER.debug(f"{tracker_settings.USE_MANUAL_MATCH=}")
        reid_weights = download_weights(cfg.reid_weights)
        
        self.device = f'cuda:{cfg.device_id}' if torch.cuda.is_available() else 'cpu'
        self.fp16 = cfg.fp16
        self.max_dist=cfg.max_dist
        self.max_iou_dist=cfg.max_iou_dist
        self.max_age=cfg.max_age
        self.n_init=cfg.n_init
        self.nn_budget=cfg.nn_budget
        self.mc_lambda=cfg.mc_lambda
        self.ema_alpha=cfg.ema_alpha
        
        self.per_class = False
        rab = ReidAutoBackend(
            weights=reid_weights, 
            device=self.device,
            half=self.fp16
        )
        self.model = rab.get_backend()
        
        self.initialize_tracker()
        
    def initialize_tracker(self):
        self.tracker = Tracker(
            metric = NearestNeighborDistanceMetric("cosine", self.max_dist, self.nn_budget),
            max_iou_dist=self.max_iou_dist,
            max_age=self.max_age,
            n_init=self.n_init,
            mc_lambda=self.mc_lambda,
            ema_alpha=self.ema_alpha
        )
        
        self.following = False
        self.followed_track_id = -1
        self.frames = 0
        self.miss = 0
        self.hits = 0
        
        # FOR LOGS
        self.previous_nb_of_persons = -1
        self.previous_followed_in_frame = False
        self.previous_max_cost_matrix = -1
        
        # for debugging
        self.frames_saved = 0
    
    def update(self, bboxes, img, reinit=False):
        """
        Before updating the tracker , compute the embedddings of each bbox
        and also compute the IOU between the bboxes

        Args:
            bboxes (Bbox): bboxes received from the object detection model
            img (np.array): the frame containing the objects detected
            reinit (bool): restart the tracker
        """
        if reinit:
            self.initialize_tracker()
            LOGGER.debug(f"reinitialized")
        
        if self.frames == 0:
            self.height, self.width = img.shape[:2]
        self.frames +=1
        
        for bbox in bboxes:
            self.narrow_bbox(bbox)
            bbox.emb = self.model.get_features(bbox.xyxy_narrowed(), img)
        
        tracks, match_data = self.tracker.update(bboxes)    
        
        if not self.following:
            self.check_same_ids(tracks)
            last_track_bboxes = [track.bbox_info for track in tracks]
            self.initiate_followed_id(last_track_bboxes)
        else:
            self.update_followed_id(tracks)
        
        self.prepare_msg_to_publish(bboxes)
            
        self.check_save_img(img, tracks)
        self.logs(bboxes, tracks, match_data)
        
        to_publish = True
        if self.miss > 0 and self.miss <= 5:
            to_publish = False
        return self.msg, to_publish
    
    def initiate_followed_id(self, bboxes):
        
        selected_bbox, index = self.choose_id(bboxes)
        
        if selected_bbox is not None:
            self.following = True
            self.set_follow_track(index)
            self.followed_track_id = index
            self.hits = 1
        LOGGER.debug(f"{selected_bbox=} and {index=}")
        self.bbox = selected_bbox
            
    def update_followed_id(self, tracks):
        ids = [track.id for track in tracks if track.in_frame]
        if self.followed_track_id in ids:
            self.hits += 1
            self.miss = 0
            self.last_update = time.time()
            index = ids.index(self.followed_track_id)
            self.bbox = tracks[index].bbox_info
        else:
            self.miss += 1
            self.bbox = None
            
    def set_follow_track(self, index):
        """Change the status of a particular track to followed"""
        if not isinstance(index, int):
            raise TypeError(f"Error with track index: {index} is not an int")    
        self.tracker.set_follow_track(index)
    
    def narrow_bbox(self, bbox):
        """
        Function to narrow the Bbox according to the data received from the pose estimation model
        """
        bbox.orig_x = bbox.x
        bbox.orig_w = bbox.w
        if bbox.keypoints:
            shoulders = keypoints.check_shoulders(bbox.keypoints)
            # shoulders = keypoints.check_ears(bbox.keypoints)
            if shoulders is not None:
                shoulders = shoulders.cpu()
                bbox.narrow_x = torch.abs((shoulders[0,0] + shoulders[1,0])//2).to(torch.int32).item()
                bbox.narrow_w = torch.abs((shoulders[0,0] - shoulders[1,0])).to(torch.int32).item()
                
            if bbox.narrow_w > THRESHOLD_SHOULDERS:
                bbox.xyxy_narrowed()
            # else:
                # LOGGER.debug("THE SHOULDERS ARE NOT FAR ENOUGH")
    
    def prepare_msg_to_publish(self,bboxes):
        self.msg = {}
        obj_found = []
        self.msg["camera_spec"] = {"width":self.width, "height":self.height}
        if self.bbox is not None:
            obj = {}
            obj["id"] = int(self.followed_track_id) if self.followed_track_id is not None else 200
            
            obj["posn"] = {"x":int(self.bbox.x if self.bbox.orig_x==-1 else self.bbox.orig_x),
                           "y":int(self.bbox.y),
                           "w":int(self.bbox.w if self.bbox.orig_w==-1 else self.bbox.orig_w),
                           "h":int(self.bbox.h)}
            obj_found.append(obj)
        self.msg["obj_found"] = obj_found
        if follow_settings.SHOW_YOLO_DETECTION:
            self.msg["debug"] = mqtt_for_rock(bboxes,track=True,for_debug=True,id_followed=self.followed_track_id)
        
    def check_same_ids(self, tracks):
        for track in tracks:
            if track.id != track.bbox_info.id:
                raise ValueError("The IDs are not the SAME")
        """_summary_
        """    
    def check_save_img(self, img, tracks):
        for track in tracks: 
            if track.emb_added and (os.getenv('DEBUG', 'False') == 'True'):
                self.save_image(img,track)
    
    def save_image(self, img, track):
        folder_path = follow_settings.DETECTIONS_FOLDER_PATH
        if self.frames_saved == 0:
            if os.path.exists(folder_path):
                # Loop through the files in the directory and delete each file
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    
                    # Check if it's a file (not a directory)
                    if os.path.isfile(file_path) and file_path.endswith(".jpg"):
                        os.remove(file_path)
            else:
                os.makedirs(folder_path) 
                
        x1, y1, x2, y2 = track.bbox_info.xyxy
        eyes = keypoints.check_eyes(track.bbox_info.keypoints)
        ears = keypoints.check_ears(track.bbox_info.keypoints)
        points = []
        if eyes is not None:
            points.extend(eyes.cpu().numpy().astype(int))
        if ears is not None:
            points.extend(ears.cpu().numpy().astype(int))
                
        # Define the circle parameters
        radius = 3  # Radius of the point (circle)
        color = (0, 0, 255)  # Red color in BGR format
        thickness = -1  # Thickness -1 means the circle will be filled

        if points:
            for point in points:
            # Draw the point (small circle) on the image
                cv2.circle(img, tuple(point), radius, color, thickness)
        # LOGGER.debug(f"SAVE IMAGE: {track.id=} and {[x1,y1,x2,y2]=}")
        text = f"w:{track.bbox_info.w}"
        cv2.putText(img, text, (x1,y2), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
        # cv2.rectangle(img, (x1+25,y1), (x2+25,y2), (255,0,0), 2)
        cv2.imwrite(f'{folder_path}/saved_image{track.hits-1}_id_{track.id}.jpg', img)
        self.frames_saved += 1
        
        
    def logs(self, bboxes_yolo, tracks, match_data:MatchData):
        string_to_log = ""
        nb_of_detections = len(bboxes_yolo)
        new_count = False
        new_follow = False
        cost_matrix_log = match_data.cost_matrix
        iou_cost = match_data.iou_cost
        appearance_cost = match_data.appearance_cost
        
        
        
        # if nb_of_detections != self.previous_nb_of_persons:
        #     new_count = True
        #     string_to_log += f"$$$ NEW COUNT: {nb_of_detections} persons\n"
            
        followed_in_frame = False
        misses = -1
        for track in tracks:
            if track.id == self.followed_track_id:
                misses = track.miss
            if track.is_followed and track.in_frame:
                followed_in_frame = True
                break
        
        if followed_in_frame != self.previous_followed_in_frame  or new_count:
            if followed_in_frame == True:
                new_follow = True
            string_to_log += f'Followed in frame: {followed_in_frame} '
            if not followed_in_frame:
                string_to_log += f"detections= {nb_of_detections}, track_ids={[track.id for track in tracks]} appearance={appearance_cost}  positional={iou_cost} followed_miss={misses} "

        if len(cost_matrix_log) > 0:
            for i,track in enumerate(tracks):
                if tracks[i].is_followed:
                    index = np.argmin(cost_matrix_log[i,:])
                    if cost_matrix_log[i,index] > self.previous_max_cost_matrix:
                        # string_to_log += f"adequation with followed ID: {cost_matrix_log[i,:]} and argmin {np.argmin(cost_matrix_log[i,:])} "
                        self.previous_max_cost_matrix = cost_matrix_log[i,index]
                    if new_count or new_follow:
                        string_to_log += f"track followed index: {i} COST MATRIX: {cost_matrix_log[i,:]}"
                    
        if len(string_to_log) > 0:
            string_to_log += "\n END_FRAME"
            LOGGER.debug(f"\n{get_time()}: {string_to_log}")
            
        self.previous_nb_of_persons = nb_of_detections
        self.previous_followed_in_frame = followed_in_frame

class BetsTracker(BaseTracker):
    """
    Tracker written by Bets that uses only the IOU
    """
    pass
    