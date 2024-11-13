from time import time
from dataclasses import dataclass, field

import numpy as np

from img_xtend.utils import LOGGER, get_time, log_to_file
from img_xtend.detection.bbox import Bbox
from img_xtend.pose_estimation import keypoints
from img_xtend.tracker import tracker_settings
from img_xtend.pipelines.follow_person import follow_settings


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3

@dataclass
class TrackData:
    last_time_embds_update: float
    last_time_seen: float
    bbox: Bbox          # The bbox of the last time was detected
    
    id: int = -1        # The ID of the track. initialized when creating a new track
    hits: int = 1
    miss: int = 0
    in_last_frame: bool = True
    state: TrackState = TrackState.Confirmed
    track_followed: bool = False
    embds: np.ndarray = np.empty((10,512))  # The array containing different embedding's detections of the track
    nb_of_embds: int = 0
    
    emb_added: bool = False
    
    def add_embedding(self, embd):
        if self.nb_of_embds < 10:
            self.embds[self.nb_of_embds,:] = embd
            self.nb_of_embds += 1
        else:
            pass
    

class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """
    def __init__(
        self,
        detection:Bbox,
        id,
        n_init,
        max_age,
        ema_alpha,
        track_followed
    ):
        self.track_data = TrackData(id=id,
                                    last_time_embds_update=time(),
                                    last_time_seen=time(),
                                    track_followed=track_followed,
                                    bbox=detection)
        if detection.emb is not None:
            embd = detection.emb/np.linalg.norm(detection.emb)
            self.track_data.add_embedding(embd=embd)
        
        self.bbox_info = detection
        self.id = id
        self.bbox_info.id = self.id
        # self.bbox = detection.to_xyah()
        self.bbox = detection.xyxy
        self.conf = detection.conf
        self.cls = detection.cls
        # self.det_ind = detection.det_ind
        self.hits = 1
        self.age = 1
        self.time_since_update = 1
        self.last_time_update = time()
        self.ema_alpha = ema_alpha
        self.n_init = n_init
        self._max_age = max_age
        self.miss = 0
        self.last_time_seen = time()
        self.in_frame = True
        self.emb_added: bool = True
        
        
        self.state = TrackState.Confirmed
        self.track_followed=track_followed
        self.features = []
        self.similarity_detection = []  # list containing the most similar detection to the new detection
        bbox_state = keypoints.check_person_state(self.bbox_info.keypoints)
        self.features_state = [bbox_state] # list containing the position (face, back, profile) of the features saved
        self.detection_from=['first']
        if detection.emb is not None:
            detection.emb /= np.linalg.norm(detection.emb)
            self.features.append(detection.emb)
            self.similarity_detection.append(self.hits)
        
        self.features_faces = []
        self.features_back = []
        self.features_profile = []
        
        # self.kf = KalmanFilter()
        # self.mean, self.covariance = self.kf.initiate(self.bbox)

    def __repr__(self):
        return f"[id={self.id}, hits={self.hits}, miss={self.miss}, bbox={self.bbox_info},\
state={self.state}, features_len={len(self.features)}, track_followed={self.track_followed},\
in_frame={self.in_frame}]\n" 
    
    def to_tlwh(self):
        w = self.bbox_info.w
        h = self.bbox_info.h
        x = self.bbox_info.x - w//2
        y = self.bbox_info.y - h//2
        ret = np.array([x,y,w,h])
        return ret
    
    def update(self,
               bbox:Bbox,
               match_data,
               similarity_index=-1,
               add_embedding=False,
               similarity_val = -100,
               matches_from= None
    ):
        # update son State
        # update son bbox, cls, 
        # LOGGER.debug(f"{person_state=}")
        self.emb_added = False   
        
        bbox_state = keypoints.check_person_state(bbox.keypoints)
        # LOGGER.debug(f"{bbox_state = }")
        
        if time() - self.last_time_update > 0.5 and add_embedding:
            # LOGGER.debug(f"{similarity_index=}")
            if self.check_valid_embedding(bbox):
                try:
                    if bbox_state != self.features_state[int(similarity_index)]:
                        LOGGER.debug("DIFFERENT PERSON STATE BETWEEN THE FEATURES")
                except:
                    LOGGER.debug("NO FEATURE STATE INDEX")
                    
                sentence_to_log = f"\n{get_time()}: {matches_from}: new features to followed ID_{self.id}  hits= {self.hits} similarity with {self.similarity_detection[int(similarity_index)]} and similiraty_val={similarity_val:.3f}"
                
                log_to_file(follow_settings.NEW_DETECTION_INFO_FILE,f"{sentence_to_log} \n")
                if matches_from in ['IOU']:
                    log_to_file(follow_settings.NEW_DETECTION_INFO_FILE,
                                f"appearance_cost: {match_data.appearance_cost} and bboxes: {match_data.bboxes} \n")
                    
                # LOGGER.debug(sentence_to_log)
                
                log_to_file(follow_settings.NEW_DETECTION_INFO_FILE,f"hits and positions={[(i,j, k) for i,j,k in zip(self.similarity_detection, self.features_state, self.detection_from)]} \n ")
                
                added = False
                if len(self.features) > tracker_settings.NB_OF_FEATURES_VECTORS:
                    for i,feats in enumerate(self.features[1:]):
                        index = i+1
                        if self.features_state[index] == bbox_state:
                            self.features[index] = bbox.emb/np.linalg.norm(bbox.emb)
                            self.similarity_detection[index] = self.hits
                            self.features_state[index] = bbox_state
                            self.detection_from[index] = matches_from
                            added = True
                            break 
                if not added:
                    if len(self.features) > 30:
                        self.features = [self.features[0]] + self.features[2:]
                        self.similarity_detection = [self.similarity_detection[0]] + self.similarity_detection[2:]
                        self.features_state = [self.features_state[0]] + self.features_state[2:]
                        self.detection_from = [self.detection_from[0]] + self.detection_from[2:]
                    self.features.append(bbox.emb/np.linalg.norm(bbox.emb))
                    self.similarity_detection.append(self.hits)
                    self.features_state.append(bbox_state)
                    self.detection_from.append(matches_from)
                self.last_time_update = time()
                self.emb_added = True    
        self.hits += 1
        self.miss = 0
        self.last_time_seen = time()
        self.bbox_info = bbox
        self.bbox_info.id = self.id
        self.in_frame = True
        
        self.track_data.hits += 1
        self.track_data.miss = 0
        self.track_data.in_last_frame = True
        self.track_data.bbox = bbox
        self.track_data.last_time_seen = time()
        self.track_data.add_embedding(bbox.emb)
    
    def change_follow_status(self):
        self.track_followed = True
        LOGGER.debug(f'$$$ TRACK FOLLOWED: {self.id}   $$$')
        
    def check_valid_embedding(self, bbox:Bbox) -> bool:
        from .check_embedding import check_embedding
        return check_embedding(bbox)
    
    def increment_age(self, features):
        self.age += 1
        time_since_update = time() - self.last_time_update
        if time_since_update > 1:
            self.features = self.features[1:] + features
        self.last_time_update = time()
        
    
    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        self.miss += 1
        self.emb_added = False
        self.in_frame = False
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif not self.track_followed and self.miss > self._max_age :
            # Delete only if it's not followed
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
    
    @property
    def is_followed(self):
        return self.track_followed