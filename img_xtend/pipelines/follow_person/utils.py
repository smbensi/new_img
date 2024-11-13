from typing import List


from img_xtend.utils import LOGGER
from img_xtend.detection.bbox import Bbox

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