import os
import abc
from typing import List
import time
import json

import torch
import numpy as np
import cv2

from img_xtend.utils import LOGGER, get_time
from img_xtend.settings import integration_stg
from img_xtend.mqtt import mqtt_handler, mqtt_settings


from img_xtend.detection.bbox import Bbox
from img_xtend.pose_estimation import keypoints
from img_xtend.tracker.config.config_utils import weights_path, config_path, get_config
from img_xtend.tracker import tracker_settings, tracker
from img_xtend.tracker.matching.metrics_matching import NearestNeighborDistanceMetric
from img_xtend.tracker.matching.match_dataclass import MatchData


from img_xtend.pipelines.follow_person import follow_settings, utils

def on_message(client, userdata, message):

    decoded_message = (message.payload.decode("utf-8"))
    LOGGER.debug(f"topic={message.topic} msg={decoded_message}")
    
    
    if message.topic == userdata["object_detection"]["TOPICS_FROM_BRAIN"]["IS_ALIVE"]:
        client.publish(userdata["object_detection"]["TOPICS_TO_BRAIN"]["APP_ALIVE"],
                       json.dumps(userdata["object_detection"]["msg_alive"]))

    elif message.topic == userdata["object_detection"]["TOPICS_TO_BRAIN"]["APP_ALIVE"]:
        LOGGER.debug(f"APP ALIVE {json.loads(decoded_message)}")
        
    # elif message.topic == userdata["object_detection"]["TOPICS_TO_BRAIN"]["DETECTION_FDBK"]:
    #     # print(decoded_message)
    #     pass
    
    # elif message.topic == userdata["object_detection"]["TOPICS_TO_ROS"]["BBOX_OBJECT"]:
    #     # LOGGER.debug(f"TO ROS: {decoded_message}")
    #     pass
        
    
    elif message.topic == userdata["object_detection"]["TOPICS_FROM_BRAIN"]["ACTIVATE_FOLLOW_ME"]:
        LOGGER.debug(f"msg for FOLLOW topic : {decoded_message} and type= {type(decoded_message)}")
        if isinstance(decoded_message, str):
            LOGGER.debug("the message is a string")
            
        if decoded_message == "True":
            integration_stg.RUN_TRACKING = True
            follow_settings.INIT_TRACKER = True
            follow_settings.MATCHES_SOURCE = {"FRAMES":0, "IOU":0, "APPEARANCE":0,"MANUAL":0}
            
        elif decoded_message == "False":
            integration_stg.RUN_TRACKING = False
            manual_pourcentage = 100* follow_settings.MATCHES_SOURCE["MANUAL"] / follow_settings.MATCHES_SOURCE["FRAMES"]
            
            appearance_pourcentage = 100* follow_settings.MATCHES_SOURCE["APPEARANCE"] / follow_settings.MATCHES_SOURCE["FRAMES"]
            
            iou_pourcentage = 100* follow_settings.MATCHES_SOURCE["IOU"] / follow_settings.MATCHES_SOURCE["FRAMES"] 
            
            LOGGER.debug(f"STATS: {follow_settings.MATCHES_SOURCE} \n Appearance={appearance_pourcentage:.0f}%  IOU={iou_pourcentage:.0f}% MANUAL={manual_pourcentage:.0f}%")



class BaseTracker(abc.ABC):
    def __init__(self):
        
        self.width = None
        self.height = None
        self.following = False
    

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
    def __init__(self, reid_model,config=None) -> None:
        """
        The constructors does 2 things:
            - Loads the Re-id model
            - Initiates the Tracker
        """
        
        super().__init__()
        tracker_type = tracker_settings.tracker_name
        tracker_cfg = get_config(tracker_type, config_path)
        LOGGER.debug(f"{follow_settings.MATCHES_TO_PRINT=}")
        LOGGER.debug(f"\n")
        LOGGER.debug(f"CONFIG TRACKER: {config_path=} {tracker_cfg}")
        LOGGER.debug(f"\n")
        LOGGER.debug(f"{tracker_settings.USE_MANUAL_MATCH=}")
        
        self.device = f'cuda:{tracker_cfg.device_id}' if torch.cuda.is_available() else 'cpu'
        self.fp16 = tracker_cfg.fp16
        self.max_dist=tracker_cfg.max_dist
        self.max_iou_dist=tracker_cfg.max_iou_dist
        self.max_age=tracker_cfg.max_age
        self.n_init=tracker_cfg.n_init
        self.nn_budget=tracker_cfg.nn_budget
        self.mc_lambda=tracker_cfg.mc_lambda
        self.ema_alpha=tracker_cfg.ema_alpha
        
        self.per_class = False
        
        self.model = reid_model
        
        self.initialize_tracker()
        
        # connect to mqtt
        self.use_mqtt = config.get("INTEGRATION",False).get("use_mqtt",False)
        if self.use_mqtt:
            topics_to_sub = [(topic,2) for topic in config["object_detection"]["TOPICS_FROM_BRAIN"].values()]
            topics_to_sub += [(topic,2) for topic in config["object_detection"]["TOPICS_TO_BRAIN"].values()]
            topics_to_sub += [(topic,2) for topic in config["object_detection"]["TOPICS_TO_ROS"].values()]
            self.mqtt_client = mqtt_handler.init_mqtt_connection('follow_me_client', topics_to_sub, config, on_message)
          
        
    def initialize_tracker(self):
        self.tracker = tracker.Tracker(
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
    
    def update(self, img: np.ndarray, bboxes):
        """
        Before updating the tracker , compute the embedddings of each bbox
        and also compute the IOU between the bboxes

        Args:
            bboxes (Bbox): bboxes received from the object detection model
            img (np.array): the frame containing the objects detected
            reinit (bool): restart the tracker
        """
        if follow_settings.INIT_TRACKER:
            self.initialize_tracker()
            LOGGER.debug(f"reinitialized")
            follow_settings.INIT_TRACKER = False
        
        if self.frames == 0:
            self.height, self.width = img.shape[:2]
        self.frames +=1
        
        for bbox in bboxes:
            self.narrow_bbox(bbox)
            crop = utils.get_crops(bbox.xyxy_narrowed(), img)
            bbox.emb = self.model.get_features(crop)
        
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
        
        selected_bbox, index = utils.choose_id(bboxes)
        
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
                
            # if bbox.narrow_w > THRESHOLD_SHOULDERS:
            #     bbox.xyxy_narrowed()
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
        # if follow_settings.SHOW_YOLO_DETECTION:
            # self.msg["debug"] = mqtt_for_rock(bboxes,track=True,for_debug=True,id_followed=self.followed_track_id)
        
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