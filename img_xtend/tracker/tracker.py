from typing import List

import numpy as np

from img_xtend.utils import LOGGER

from .track import Track

class Tracker:
    """
    Class responsible for keeping the tracks created and compute the matching algo between new bboxes in the present frame and the tracks created
    """
    def __init__(
        self,
        metric,
        max_iou_dist=0.9,
        max_age=30,
        n_init=3,
        _lambda=0,
        ema_alpha=0.9,
        mc_lambda=0.995
    ):
        self.metric = metric
        self.max_iou_dist = max_iou_dist
        self.max_age = max_age
        self.n_init = n_init
        self._lambda = _lambda
        self.ema_alpha = ema_alpha
        self.mc_lambda = mc_lambda
        
        self.tracks: List[Track] = []
        self._next_id = 1
        
    def update(self, bboxes):
        pass
    
    def _match(self, bboxes):
        appearance_match = None
        iou_match = None
        pass
        
    def _initiate_track(self, bbox):
        """If a bbox in the actual frame is not matched with previous frames, create a new track"""
        self.tracks.append(
            Track(
                bbox,
                self._next_id,
                self.n_init,
                self.max_age,
                self.ema_alpha,
                followed_id=False
            )
        )
        LOGGER.debug(f'Create new track {self._next_id}')
        self._next_id += 1
        
    def set_follow_track(self, index:int):
        self.tracks[index].change_follow_status()
        