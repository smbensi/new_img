import typing as T


from dataclasses import dataclass,field

import numpy as np
import torch

@dataclass
class Bbox:
    x: int = 0  # x-coordinate of the center of the bbox
    y: int = 0  # y-coordinate of the center of the bbox
    w: int = 0
    h: int = 0
    id: int = -1
    conf: int = 0
    cls: int = 0
    xyxy: list = field(default_factory=list)
    tlwh: list = field(default_factory=list)
    emb: np.ndarray =  np.zeros((1,512))
    height_frame: int = -1
    width_frame: int = -1
    keypoints: T.Any  = None
    orig_x: int = -1
    orig_w: int  = -1
    narrow_x:int = -1
    narrow_w: int = -1
    
    def __post_init__(self):
        # logger.debug(f"{[self.x,self.y,self.w,self.h]=}")
        self.xyxy = xywh2xyxy(np.array([self.x,self.y,self.w,self.h]))
        self.tlwh = self.to_tlwh()
    
    def __eq__(self, other):
        if not isinstance(other, Bbox):
            return False
        return self.x == other.x and \
                self.y == other.y and \
                self.w == other.w and \
                self.h == other.h

    def to_tlwh(self):
        w = self.w
        h = self.h
        x = self.x - w//2
        y = self.y - h//2
        ret = np.array([x,y,w,h])
        return ret

    def xyxy_narrowed(self):
        if self.narrow_w != -1:
            return xywh2xyxy(np.array([self.narrow_x,self.y,self.narrow_w,self.h]))
        else:
            return self.xyxy
            
    def __repr__(self):
        return f'Bbox: id={self.id}, x={self.x}, y={self.y}, w={self.w}, h={self.h}, xyxy={self.xyxy} '


def to_tlbr(bbox:Bbox):
    return np.array([bbox.x-bbox.w//2, bbox.y-bbox.h//2, bbox.w, bbox.h])       

def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    # if isinstance(y, np.ndarray):
    #     y = y.astype(np.int32)
    return y

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    # assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    # if isinstance(y, np.ndarray):
    #     y = y.astype(np.int32)
    return y