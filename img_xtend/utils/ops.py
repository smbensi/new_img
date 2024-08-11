# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py

import contextlib
import math
import re
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from img_xtend.utils import LOGGER



class Profile(contextlib.ContextDecorator):
    """
    Profile class. Use a decorator with @Profile() or as a context manager with 'with Profile():'
    """
    
    def __init__(self, t=0.0, device: torch.device = None):
        """
        Initialize the Profile class
        
        Args:
            t (float): Initial time. Defaults to 0.0
            device (torch.device): Devices used for model inference. Defaults to None (cpu)
        """
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))
    
    def __enter__(self):
        """Start timing"""
        self.start = time.time()
        return self
    
    def __exit__(self, type, value, traceback): # noqa
        """Stop timing"""
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt
        
    def __str__(self) -> str:
        """Returns a human readable string representing the accumulated elapsed time in the profiler"""
        return f"Elapsed time is {self.t} s"
    
    def time(self):
        """Get current time"""
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.time()
    
def clean_str(s):
    """
    Cleans a string by replacing special characters with underscore _

    Args:
        s (str): a string needing special characters replaced

    Returns:
        (str): a string with special characters replaced by an underscore _
    """
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)