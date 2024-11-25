from typing import List

import cv2
import numpy as np
import torch

from img_xtend.utils import LOGGER
from img_xtend.detection.bbox import Bbox


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


def choose_id(bboxes:List[Bbox]):
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


def get_crops(xyxys, img):
        # if xyxys.shape[0] == 1:
        #     xyxys = [xyxys]
        crops = []
        h, w = img.shape[:2]
        resize_dims = (128, 256)
        interpolation_method = cv2.INTER_LINEAR
        mean_array = np.array([0.485, 0.456, 0.406])
        std_array = np.array([0.229, 0.224, 0.225])
        # dets are of different sizes so batch preprocessing is not possible
        # for box in xyxys:
        x1, y1, x2, y2 = xyxys.astype('int')
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)
        crop = img[y1:y2, x1:x2]
        # resize
        crop = cv2.resize(
            crop,
            resize_dims,  # from (x, y) to (128, 256) | (w, h)
            interpolation=interpolation_method,
        )

        # (cv2) BGR 2 (PIL) RGB. The ReID models have been trained with this channel order
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = np.expand_dims(crop, axis=0)
        # crop = torch.from_numpy(crop).float()
        # crops.append(crop)

        # List of torch tensor crops to unified torch tensor
        # crops = torch.stack(crops, dim=0)

        # Normalize the batch
        # crops = crops / 255.0
        crops = crop / 255.0

        # Standardize the batch
        crops = (crops - mean_array) / std_array

        crops = np.transpose(crops, axes=(0,3,1,2))
        # crops = torch.permute(crops, (0, 3, 1, 2))
        # crops = crops.to(dtype=torch.half if self.half else torch.float, device=self.device)

        return crops