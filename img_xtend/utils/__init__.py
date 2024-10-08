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
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"



def set_logging(name="LOGGING_NAME", verbose=True):
    """Sets up logging for the given name with UTF-8 encoding support, ensuring compatibility across different platforms"""
    level = logging.DEBUG if verbose and RANK in {-1, 0} else logging.ERROR # rank in world for Multi-GPU trainings
    
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

ONLINE = is_online()


class IterableSimpleNamespace(SimpleNamespace):
    """It's an extension class of SimpleNamespace that adds iterable functionality and enables usage with dict() and for loops"""
    def __iter__(self):
        """Returns an iterator of ke-value pairs from the namespace's attributes"""
        return iter(vars(self).items())
    
    def __str__(self):
        """Returns a human-readable string representation of the object"""
        return  "\n".join(f"{k}={v}" for k, v in vars(self).items())
    
    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(
            f"""
            '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics
            'default.yaml' file.\nPlease update your code with 'pip install -U ultralytics' and if necessary replace
            {DEFAULT_CFG_PATH} with the latest version from
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
            """
        )
        
    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)



def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file

    Args:
        file (str, optional): filename. Defaults to "data.yaml".
        append_filename (bool, optional):add the yaml filename to the YAML dictionary. Defaults to False.
        
    Returns:
        (dict): YAML data and filename
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file)
        return data
    
    
DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
for k, v in DEFAULT_CFG_DICT.items():
    if isinstance(v, str) and v.lower() == "none":
        DEFAULT_CFG_DICT[k] = None
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)    


def get_default_args(func):
    """
    Returns a dictionary of defaults arguments for a function

    Args:
        func (callable): the function to inspect
    
    Returns:
        (dict):  A dictionary where each key is a parameter name, and each value is the default value of that parameter
    """
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def colorstr(*input):
    """
    Colors a string based on the provided color and style arguments. Utilizes ANSI escape codes.
    See https://en.wikipedia.org/wiki/ANSI_escape_code for more details.

    This function can be called in two ways:
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    In the second form, 'blue' and 'bold' will be applied by default.

    Args:
        *input (str): A sequence of strings where the first n-1 strings are color and style arguments,
                      and the last string is the one to be colored.

    Supported Colors and Styles:
        Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        Misc: 'end', 'bold', 'underline'

    Returns:
        (str): The input string wrapped with ANSI escape codes for the specified color and style.

    Examples:
        >>> colorstr("blue", "bold", "hello world")
        >>> "\033[34m\033[1mhello world\033[0m"
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


class TryExcept(contextlib.ContextDecorator):
    """
    Ultralytics TryExcept class. Use as @TryExcept() decorator or 'with TryExcept():' context manager.

    Examples:
        As a decorator:
        >>> @TryExcept(msg="Error occurred in func", verbose=True)
        >>> def func():
        >>>    # Function logic here
        >>>     pass

        As a context manager:
        >>> with TryExcept(msg="Error occurred in block", verbose=True):
        >>>     # Code block here
        >>>     pass
    """

    def __init__(self, msg="", verbose=True):
        """Initialize TryExcept class with optional message and verbosity settings."""
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        """Executes when entering TryExcept context, initializes instance."""
        pass

    def __exit__(self, exc_type, value, traceback):
        """Defines behavior when exiting a 'with' block, prints error message if necessary."""
        if self.verbose and value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True