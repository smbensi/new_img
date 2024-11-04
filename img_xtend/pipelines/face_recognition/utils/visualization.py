from copy import deepcopy

import torch 
import cv2

from img_xtend.pose_estimation.keypoints import *

def show_pose(img, pose_result):
    img_new = deepcopy(img)
    # Define the color (BGR format) and radius of the point
    color = (0, 0, 255)  # Red color in BGR format
    radius = 5  # Radius of the circle (point size)
    thickness = -1  # -1 fills the circle to make it a solid dot
    
    pose_keypoint = pose_result.keypoints
    eyes = check_eyes(pose_keypoint)
    nose = check_nose(pose_keypoint)
    points = []
    # joint_points = [eyes
    if isinstance(eyes, torch.Tensor):
        for i in range(eyes.shape[0]):
            points.append((eyes[i,0].item(), eyes[i,1].item()))
    if isinstance(nose, torch.Tensor):
        points.append((nose[0].item(), nose[1].item()))
        
    for point in points:
        cv2.circle(img_new, point, radius, color, thickness)
    return img_new
