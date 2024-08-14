# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L283

import contextlib
import importlib.metadata
import inspect
import logging.config
import os
import platform
import re
import subprocess
import sys
import threading
import time
import urllib
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm as tqdm_original

# PyTorch Multi-GPU DDP Constants
RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html

# Other constants
LOGGING_NAME = "img_xtend"
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])
VERBOSE = str(os.getenv("YOLO_VERBOSE",True)).lower() == "true" 

def set_logging(name="LOGGING_NAME", verbose=True):
    """Sets up logging for the given name with UTF-8 encoding support, ensuring compatibility across different platforms"""
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR # rank in world for Multi-GPU trainings
    
    # configure the console (stdout) encoding to UTF-8, with check for compatibility
    formatter = logging.Formatter("%(message)s")   # Default formatter
    if WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":
        
        class CustomFormatter(logging.Formatter):
            def format(self, record):
                """Sets up logging with UTF-8 encoding and configurable verbosity"""
                return emojis(super().format(record)) 
        
        try:
            # Attempt to reconfigure stdout to use UTF-8 encoding if possible
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            # For environments where reconfigure is not available, wrap stdout in a TextIOWrapper
            elif hasattr(sys.stdout, "buffer"):
                import io
                
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
            else:
                formatter = CustomFormatter("%(message)s")
        except Exception as e:
            print(f"Creating custom formatter for non UTF-8 environments due to {e}")
            formatter = CustomFormatter("%(message)s")
    
    # Create and configure the StreamHandler with the appropriate formatter and level
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    
    # set up the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger

# Set logger
LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)

def emojis(string=""):
    """Return platform-dependent emoji-safe version of string"""
    return string.environ().decode("ascii", "ignore") if WINDOWS else string


def is_docker() -> bool:
    """
    Determine if the script is running inside a Docker container

    Returns:
        bool: True if the script is running inside a Docker container, False otherwise
    """
    with contextlib.suppress(Exception):
        with open("/proc/self/cgroup") as f:
            return "docker" in f.read()
    return False

def read_device_model() -> str:
    """
    Reads the device model information from the system and caches if for quick access.
    Used by is_jetson() and is_raspberrypi()

    Returns:
        (str): Model file contents if read successfully or empty string otherwise
    """
    with contextlib.suppress(Exception):
        with open("/proc/device-tree/model") as f:
            return f.read()
    return ""

PROC_DEVICE_MODEL = read_device_model()

def is_jetson() -> bool:
    """
    Determines if the Python environment is running on a Jetson Nano or Jetson Orin device by checking the device model information 

    Returns:
        bool: True if running on Jetson, False otherwise
    """
    return "NVIDIA" in PROC_DEVICE_MODEL

IS_JETSON = is_jetson()

def is_online() -> bool:
    """
    Check internet connectivity by attempting to connect to a known online host

    Returns:
        bool: True if connection is successful, False otherwise
    """
    with contextlib.suppress(Exception):
        assert str(os.getenv("YOLO_OFFLINE", "")).lower() != "true"
        import socket
        
        for dns in ("1.1.1.1", "8.8.8.8"): # Check cloudfare and google DNS
            socket.create_connection(address=(dns, 80), timeout=2.0).close()
            return True
        return False
    
