# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/checks.py

import os
import math
from pathlib import Path
import subprocess
import time
import glob

import cv2
import numpy as np
import torch

from img_xtend.utils import (
    LOGGER,
    LINUX,
    ONLINE,
    ROOT,
    colorstr,
    TryExcept
)

def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    """
    Verify image size is a multiple of the given stride in each dimension, If the image size is not a multiple of the stride, update it to the nearest multiple of the stride that is greater than or equal to the given floor value

    Args:
        imgsz (int | cList[int]): Image size
        stride (int, optional): stride value. Defaults to 32.
        min_dim (int, optional): Minimum number of dimensions. Defaults to 1.
        max_dim (int, optional): Maximum number of dimensions. Defaults to 2.
        floor (int, optional): Minimum allowed value for image size. Defaults to 0.
    
    Returns:
        (List[int]): updated image size
    """
    # Convert stride to integer if it is a tensor
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)
    
    # Convert image size to list if it is an integer
    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    elif isinstance(imgsz, str): # i.e. '640' or '[640,640]'
        imgsz = [int(imgsz)] if imgsz.isnumeric() else eval(imgsz)
    else:
        raise TypeError(
            f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}. "
            f"Valid imgsz types are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'"
        )
    
    # Apply max_dim
    if len(imgsz) > max_dim:
        msg = (
            "'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list "
            "or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'"
        )
        if max_dim != 1:
            raise ValueError(f"imgsz={imgsz} is not a valid image size. {msg}")
        LOGGER.warning(f"WARNING ⚠️ updating to 'imgsz={max(imgsz)}'. {msg}")
        imgsz = [max(imgsz)]
    # Make image size a multiple of the stride
    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]
    
    # Print warning message if image size was updated
    if sz != imgsz:
        LOGGER.warning(f"WARNING ⚠️ imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}")

    # Add missing dimensions if necessary
    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz

    return sz

def check_file(file, suffix="", download=True, download_dir=".", hard=True):
    """Search/download file (if necessary) and return path."""
    check_suffix(file, suffix) # optional
    file = str(file).strip() # convert to string and strip spaces
    if (
        not file
        or ("://" not in file and Path(file).exists())
        or file.lower().startswith("grpc://")
    ): # file exists or gRPC Triton images
        return file
    elif download and file.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")): # download
        url = file
        file = Path(download_dir) / url2file(file)
        if file.exists():
            LOGGER.info(f"Found {clean_url(url)} locally at {file}") # file already exists
        else:
            downloads.safe_download(url=url, file=file, unzip=False)
        return str(file)
    else: # search
        files = glob.glob(str(ROOT / "**" / file), recursive=True) or glob.glob(str(ROOT.parent / file)) # find file
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
        return files[0] if len(files) else [] # return file


def check_yaml(file, suffix=(".yaml",".yml"), hard=True):
    """Search/download YAML file (if necessary) and return path, checking suffix"""
    return check_file(file, suffix, hard=hard)

def check_imshow(warn=False):
    """Check if environment supports image displays"""
    try:
        if LINUX:
            assert "DISPLAY" in os.environ, "The DISPLAY environment variable isn't set."
        cv2.imshow("test", np.zeros((8,8,3), dtype=np.uint8)) # show a small 8-pixel image
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f"WARNING ⚠️ Environment does not support cv2.imshow() or PIL Image.show()\n{e}")
        return False
    
    
def check_suffix(file="yolov8n.pt", suffix=".pt", msg=""):
    """Check file(s) for acceptable suffix"""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix,)
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower().strip()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}, not {s}"
                
@TryExcept()
def check_requirements(requirements=ROOT.parent / "requirements.txt", exclude=(), install=True, cmds=""):
    """
    Check if installed dependencies meet YOLOv8 requirements and attempt to auto-update if needed.

    Args:
        requirements (Union[Path, str, List[str]]): Path to a requirements.txt file, a single package requirement as a
            string, or a list of package requirements as strings.
        exclude (Tuple[str]): Tuple of package names to exclude from checking.
        install (bool): If True, attempt to auto-update packages that don't meet requirements.
        cmds (str): Additional commands to pass to the pip install command when auto-updating.

    Example:
        ```python
        from ultralytics.utils.checks import check_requirements

        # Check a requirements.txt file
        check_requirements('path/to/requirements.txt')

        # Check a single package
        check_requirements('ultralytics>=8.0.0')

        # Check multiple packages
        check_requirements(['numpy', 'ultralytics>=8.0.0'])
        ```
    """

    prefix = colorstr("red", "bold", "requirements:")
    check_python()  # check python version
    check_torchvision()  # check torch-torchvision compatibility
    if isinstance(requirements, Path):  # requirements.txt file
        file = requirements.resolve()
        assert file.exists(), f"{prefix} {file} not found, check failed."
        requirements = [f"{x.name}{x.specifier}" for x in parse_requirements(file) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]

    pkgs = []
    for r in requirements:
        r_stripped = r.split("/")[-1].replace(".git", "")  # replace git+https://org/repo.git -> 'repo'
        match = re.match(r"([a-zA-Z0-9-_]+)([<>!=~]+.*)?", r_stripped)
        name, required = match[1], match[2].strip() if match[2] else ""
        try:
            assert check_version(metadata.version(name), required)  # exception if requirements not met
        except (AssertionError, metadata.PackageNotFoundError):
            pkgs.append(r)

    @Retry(times=2, delay=1)
    def attempt_install(packages, commands):
        """Attempt pip install command with retries on failure."""
        return subprocess.check_output(f"pip install --no-cache-dir {packages} {commands}", shell=True).decode()

    s = " ".join(f'"{x}"' for x in pkgs)  # console string
    if s:
        if install and AUTOINSTALL:  # check environment variable
            n = len(pkgs)  # number of packages updates
            LOGGER.info(f"{prefix} Ultralytics requirement{'s' * (n > 1)} {pkgs} not found, attempting AutoUpdate...")
            try:
                t = time.time()
                assert ONLINE, "AutoUpdate skipped (offline)"
                LOGGER.info(attempt_install(s, cmds))
                dt = time.time() - t
                LOGGER.info(
                    f"{prefix} AutoUpdate success ✅ {dt:.1f}s, installed {n} package{'s' * (n > 1)}: {pkgs}\n"
                    f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
                )
            except Exception as e:
                LOGGER.warning(f"{prefix} ❌ {e}")
                return False
        else:
            return False

    return True