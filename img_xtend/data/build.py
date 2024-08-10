# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/build.py#L174

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed


from .loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list
)


def check_source(source):
    """Check source type and return corresponding flag values"""
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    
    if isinstance(source, (str, Path, int)): # int for local usb camera
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS | VID_FORMATS)
        is_url = source.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        screenshot = source.lower() == "screen"
        if is_url and is_file:
            source = check_file(source) # download
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source) # convert all list elements to PIL or np arrays
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError("Unsupported image type.")
    
    return webcam, screenshot, from_img, in_memory, tensor

def load_inference_source(source=None, batch=1, vid_stride=1, buffer=False):
    """
    Loads an inference source for object detection and applies necessary transformations.
    
    Args:
        source (str, Path, Tensor, PIL.Image, np.ndarray): The input source for inference
        batch (int, optional): batch size for dataloaders. Default is 1
        vid_stride (int, optional): The frame interval for video sources. Default is 1.
        buffer (bool, optional): Determined whether stream frames will be buffered. Default is False.
        
    Returns:
        dataset (Dataset): a dataset object for the specified input source
    """
    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)
    
    # Dataloader
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif stream:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer)
    elif screenshot:
        pass
    elif from_img:
        dataset = LoadPilAndNumpy(source)
    else:
        dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride)
    
    # Attach source types to the dataset
    setattr(dataset, "source_type", source_type)
    
    return dataset
    