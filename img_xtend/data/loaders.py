import glob
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import torch
from PIL import Image

@dataclass
class SourceTypes:
    """Class to represent various types or input sources for predictions"""
    
    stream: bool = False
    screenshot: bool = False
    from_img: bool = False
    tensor: bool = False
    
class LoadStreams:
    """
    Stream loader for various types of video streams, Supports RTSP, HTTP and TCP streams.
    
    Attributes:
        sources (str): The source input paths or URLs for the video streams
        vid_stride (int): Video frame-rate stride, defaults to 1.
        buffer (bool): Whether to buffer input streams, defaults to False
        running (bool): Flag to indicate if the streaming thread is running
        mode (str): Set to 'stream' indicating real-time capture
        imgs (list): List of image frames for each stream
        fps (list): List of FPS for each stream
        frames (list): List of total frames for each stream
        threads (list): List of threads for each stream
        shape (list): list of shapes for each stream
        caps (list): list of cv2.VideoCapture objects for each stream
        bs (int): Batch size for processing
        
    Methods:
        __init__ : Initialize the stream loader
        update: Read stream frames in daemon thread
        close: Close stream loader and release resources
        __iter__ : Returns an iterator object for the class
        __next__ : Returns source paths, transformed, and original images for processing
        __len__ : Returns the length of the sources object
    """
    
    def __init__(self, sources='file.streams', vid_stride=1, buffer=False):
        """Initialize instance variables and check for consistent input stream shapes."""
        torch.backends.cudnn.benchmark = True # faster for fixed-size inference
        self.buffer = buffer # buffer input streams
        self.running = True # running flag for Thread
        self.mode = "stream"
        self.vid_stride = vid_stride # video frame-rate stride
        
        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.bs = n
        self.fps = [0] * n # frames-per-second
        self.frames = [0] * n
        self.threads = [None] * n
        self.caps = [None] * n # video capture objects
        self.imgs = [[] for _ in range(n)] # images
        self.shape = [[] for _ in range(n)] # image shapes
        self.sources = [ops.clean_str(x) for x in sources] #TODO add ops clean source names for later 
        
        for i, s in enumerate(sources): # index, source
            # Start thread to read frames from video stream
            st = f"{i+1}/{n}: {s}..."
            if urlparse(s).hostname in {"www.youtube.com", "youtube.com", "youtu.be"}: # if source is youtube video
                s = get_best_youtube_url(s) # TODO add this function
            s = eval(s) if s.isnumeric() else s
            
            self.caps[i] = cv2.VideoCapture(s) # store video capture object 
            if not self.caps[i].isOpened():
                raise ConnectionError(f"{st} failed to open {s}")
            w = int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.caps[i].get(cv2.CAP_PROP_FPS) # warning may return 0 or nan
            self.frames[i] = max(int(self.caps[i].get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float(
                "inf"
            ) # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0)
            
            success, im = self.caps[i].read() # guarantee first frame
            if not success or im is None:
                raise ConnectionError(f"{st} failed to read images from {s}")
            self.imgs[i].append(i)
            self.shape[i] = im.shape
            self.threads[i] = Thread(target=self.update, args=([i, self.caps[i], s]), daemon=True)
            LOGGER.info(f"{st}Success ✅ ({self.frame[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)") # TODO add LOGGER
            self.threads[i].start()
        LOGGER.info("")
        
    
    def update(self, i, cap, stream):
        """read stream `i` frames in daemon thread"""
        n, f = 0, self.frames[i] # frame number, frame array
        while self.running and cap.isOpened() and n < (f-1):
            if len(self.imgs[i]) < 30:     # keep a <= 30-image buffer 
                n += 1
                cap.grab()   # .read() = .grab() followed by .retrieve()
                if n % self.vid_stride == 0:
                    success, im = cap.retrieve()
                    if not success: 
                        im = np.zeros(self.shape[i], dtype=np.uint8)
                        LOGGER.warning ("WARNING video stream unresponsive, please check your IP camera")
                        cap.open(stream)  # re-open stream if signal was lost
                    if self.buffer:
                        self.imgs[i].append(im)
                    else:
                        self.imgs[i] = [im]
            
            else:
                time.sleep(0.01) # wait until the buffer is empty
    
    def close(self):
        """Close stream loader and release resources"""
        self.running = False # stop flag for Thread
        for thread in self.threads: 
            if thread.is_alive():
                thread.join(timeout=5) # add timeout
        for cap in self.caps:  # Iterate through the stored VideoCapture objects
            try:
                cap.release()
            except Exception as e:
                LOGGER.warning(f"WARNING could not relase the VideoCapture Object: {e}")
        cv2.destroyAllWindows()
        
    def __iter__(self):
        """Iterates through image feed and re-open unresponsive streams"""
        self.count = -1
        return self
    
    def __next__(self):
        """returns source paths, transformed and original images for processing"""
        self.count += 1
        
        images = []
        for i, x in enumerate(self.imgs):
            # wait until a frame is available in each buffer
            while not x:
                if not self.threads[i].is_alive() or cv2.waitKey(1) == ord('q'): # q to quit
                    self.close()
                    raise StopIteration
                time.sleep(1 / min(self.fps))
                x = self.imgs[i]
                if not x:
                    LOGGER.warning(f"WARNING waiting for stream {i}")
            
            # Get and remove the first frame from imgs buffer
            if self.buffer:
                images.append(x.pop(0))
            
            # Get the last frame and clear the rest from the imgs buffer
            else:
                images.append(x.pop[-1] if x else np.zeros(self.shape[i], dtype=np.uint8))
                x.clear()
        
        return self.sources, images, [""] * self.bs
    
    def __len__(self):
        """return the lenght of the sources objects"""
        return self.bs # 1E12 frames = 32 streams at 30 FPS for 30 years

class LoadImagesAndVideos:
    """
    YOLOv8 image/video dataloader
    
    This class manages the loading and preprocessing of image and video data for YOLOv8. It supports loading from various formats, including single image files, video files and list of image and video paths
    
    Attributes:
        files (list): List of image and video file paths
        nf (int): Total number of files (images and videos)
        video_flag (list): Flags indicating whether a file is a video (True) or an image (False)
        mode (str): current mode, 'image' or 'video'
        vid_stride (int): Stride for video frame rate, defaults to 1
        bs (int): Batch size, set to 1 for this class
        cap (cv2.VideoCapture): Video capture object for openCV
        frame (int): frame counter for video
        frames (int): Total number of frames in the video
        count (int): Counter for iteration, initialized at 0 during '__iter__()'
    
    Methods:
        _new_video (path): Create a new cv2.VideoCapture object for a given video path.
    """
    
    def __init__(self, path, batch=1, vid_stride=1):
        """Initialize the dataloader and raise FileNotFoundError id file not found"""
        parent = None
        if isinstance(path, str) and Path(path).suffix == '.txt': # *.txt file with img/vid/dir on each line 