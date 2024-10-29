from dataclasses import dataclass, field
from typing import List,Tuple

import numpy as np

from img_xtend.detection.bbox import Bbox
from img_xtend.tracker.track import Track

@dataclass
class MatchData:
    match_from: str
    cost_matrix: np.ndarray
    argmin_matrix: np.ndarray
    index_in_cost_matrix: Tuple[int, int] 
    matches_appearance: List[Tuple[int, int]]
    matches_iou: List[Tuple[int, int]]
    iou_cost: np.ndarray
    appearance_cost: np.ndarray
    bboxes: List[Bbox] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)

    