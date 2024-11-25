from typing import List,Tuple

import numpy as np

from img_xtend.detection.bbox import Bbox
from img_xtend.utils import  LOGGER
from img_xtend.tracker.matching import iou_matching
from img_xtend.pose_estimation import keypoints


def is_on_the_side(bbox:Bbox):
    '''check if the bounding box is on the side of frame'''
    x,y,w,h = bbox.tlwh
    if x <= 0 or x+w >= bbox.width_frame:
        return True
    return False    

def are_shoulders_inside(keypoints_results):
    shoulders = keypoints.check_shoulders(keypoints_results)
    if shoulders is None:
        return False
    return True

def is_in_appearance_interval(cost_matrix_element: float) -> bool:
    if cost_matrix_element > 0.1 and cost_matrix_element < 0.20:
        return True
    return False

def is_in_iou_interval(cost_matrix_element: float) -> bool:
    if cost_matrix_element > 0.01 and cost_matrix_element < 0.5:
        return True
    return False


def add_new_detection(matches_from:str,cost_matrix, bboxes: List[Bbox], track_idx:int, bbox_idx: int) -> bool:
    
    bbox = bboxes[bbox_idx]
    cost_matrix_element = cost_matrix[track_idx, bbox_idx]
    # LOGGER.debug(f"add new detection {matches_from=} {cost_matrix_element=}")
    iou_matrix = compute_iou_bboxes(bboxes)
    
    if not check_iou_in_present_frame(iou_matrix, bbox_idx):
        LOGGER.debug("check_iou_in_present_frame")
        return False
    if not are_shoulders_inside(bbox.keypoints):
        LOGGER.debug("shoulders are not inside")
        return False
    if matches_from in ['APPEARANCE', 'MANUAL']:
        return is_in_appearance_interval(cost_matrix_element)
    elif matches_from in ['IOU']:
        return is_in_iou_interval(cost_matrix_element)

def compute_iou_bboxes(bboxes: List[Bbox]):
    '''Compute the IOU between the bounding boxes in the frame'''
    if not bboxes or len(bboxes) < 2:
        return None
    
    bboxes_tlwh = np.zeros((len(bboxes), 4))
    for i in range(bboxes_tlwh.shape[0]):
        # LOGGER.debug(f'{bboxes[i].to_tlwh()=}')
        bboxes_tlwh[i] = bboxes[i].to_tlwh()
        
    iou_matrix = np.zeros((len(bboxes),len(bboxes)))
    for i in range(iou_matrix.shape[0]):
        iou_matrix[i] = iou_matching.iou(bboxes_tlwh[i], bboxes_tlwh)
    # LOGGER.debug(f"{bboxes_tlwh=}")    
    # LOGGER.debug(f"{iou_matrix=}")    
    return iou_matrix

def check_iou_in_present_frame(iou_matrix, bbox_idx):
        THRESH_IOU = 0.1
        if np.any(iou_matrix):
            bbox_line = iou_matrix[bbox_idx]
            # LOGGER.debug(f"{iou_matrix[bbox_idx]=} and {bbox_idx=}")
            if bbox_line.shape[0] > 1:
                ious = np.r_[bbox_line[:bbox_idx], bbox_line[bbox_idx+1:]]
                if np.any(ious > THRESH_IOU):
                    # LOGGER.debug(f"too much overlap {ious=}")
                    return False
        return True


def check_embedding(bbox:Bbox):
    '''Checks all the conditions to save an embedding or not'''
    if is_on_the_side(bbox):
        # LOGGER.debug("on the side")
        return False
    if bbox.keypoints != []:
        if keypoints.check_nose(bbox.keypoints) is not None and \
            keypoints.check_eyes(bbox.keypoints) is not None:
            # LOGGER.debug("Showing his face")
            return True
        if keypoints.check_nose(bbox.keypoints) is None \
            and keypoints.check_shoulders(bbox.keypoints) is not None:
            # LOGGER.debug("Showing his back")
            return True
    return False