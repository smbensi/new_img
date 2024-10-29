from dataclasses import dataclass, field
from typing import List
import time

import numpy as np

@dataclass
class KnownFace:
    _id : str
    collectionName : str
    frames : int    # nb of frames the face has been recognized
    last_seen : float   # time of the last time the face has been seen
    bbox : field(default_factory=list) 
    miss: int = 0   # nb of frames missed since the last time seen
    published: bool = False

@dataclass 
class UnknownFace:
    face_vector : np.ndarray
    frames : int    # nb of frames the face has been recognized
    last_seen : float # time of the last time the face has been seen
    bbox : field(default_factory=list) 
    index: int
    miss: int = 0   # nb of frames missed since the last time seen
    published: bool = False

    def __repr__(self):
        return f"(index={self.index}, frames={self.frames}, bbox={self.bbox}, miss={self.miss}, time_since_last_seen={time.time() - self.last_seen:.2f})"