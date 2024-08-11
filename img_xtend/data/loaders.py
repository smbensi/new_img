# SOURCE: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/loaders.py

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

from img_xtend.utils import LOGGER, ops

from .utils import IMG_FORMATS, VID_FORMATS, FORMATS_HELP_MSG


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
        self.sources = [ops.clean_str(x) for x in sources] 
        
        for i, s in enumerate(sources): # index, source
            # Start thread to read frames from video stream
            st = f"{i+1}/{n}: {s}..."
            if urlparse(s).hostname in {"www.youtube.com", "youtube.com", "youtu.be"}: # if source is youtube video
                s = get_best_youtube_url(s) # TODO add youtube function
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
            LOGGER.info(f"{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)")
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
            parent = Path(path).parent
            path = Path(path).read_text().splitlines() # list of sources
        files = []
        for p in sorted(path) if isinstance(path, (list,tuple)) else [path]:
            a = str(Path(p).absolute()) # do not use .resolve() https://github.com/ultralytics/ultralytics/issues/2912
            if '*' in a:
                files.extend(sorted(glob.glob(a, recursive=True))) # glob
            elif os.path.isdir(a):
                files.extend(sorted(glob.glob(os.path.join(a,'*.*')))) # dir
            elif os.path.isfile(a):
                files.append(a)   # files (absolute or relative to CWD)
            elif parent and (parent / p).is_file():
                files.append(str((parent / p).absolute())) # files (relative to *.txt file parent)
            else:
                raise FileNotFoundError(f"{p} does not exist")
            
        # Define files as images or videos
        images, videos = [], []
        for f in files:
            suffix = f.split('.')[-1].lower() # Get file extension without the dot and lowercase
            if suffix in IMG_FORMATS: 
                images.append(f)
            elif suffix in VID_FORMATS: 
                videos.append(f)
        ni, nv = len(images), len(videos)
        
        self.files = images + videos
        self.nf = ni + nv # nb of files
        self.ni = ni      # nb of images
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        self.vid_stride = vid_stride  # video frame-rate stride
        self.bs = batch
        if any(videos):
            self._new_video(videos[0]) # new video
        else:
            self.cap = None
        if self.nf == 0:
            raise FileNotFoundError(f"No images or videos found in {p}. {FORMATS_HELP_MSG}") 
        
    def __iter__(self):
        """Returns an iterator object for VideoStream or ImageFolder."""
        self.count = 0
        return self
    
    def __next__(self):
        """Returns the next batch of images or video frames along with their paths and metadata"""
        paths, imgs, info = [], [], []
        while len(imgs) < self.bs:
            if self.count >= self.nf: # end of file list
                if imgs:
                    return paths, imgs, info # return the last partial batch
                else:
                    raise StopIteration
            
            path = self.files[self.count]
            if self.video_flag[self.count]:
                self.mode = 'video'
                if not self.cap or not self.cap.isOpened():
                    self._new_video(path)
                
                for _ in range(self.vid_stride):
                    success = self.cap.grab()
                    if not success:
                        break # end of video or failure
                
                if success:
                    success, im0 = self.cap.retrieve()
                    if success:
                        self.frame += 1 
                        paths.append(path)
                        imgs.append(im0)
                        info.append(f"video {self.count + 1}/{self.nf} (frame {self.frame}/{self.frames}) {path}: ")
                        if self.frame == self.frames: # end of video
                            self.count += 1 
                            self.cap.release()
                
                else:
                    # Move to the next file if the current video ended or failed to open
                    self.count += 1
                    if self.cap:
                        self.cap.release()
                    if self.count < self.nf:
                        self._new_video(self.files[self.count])
            
            else:
                self.mode = "image"
                im0 = cv2.imread(path)  # BGR
                if im0 is None:
                    LOGGER.warning(f"WARNING Image read error {path}")
                else:
                    paths.append(path)
                    imgs.append(im0)
                    info.append(f"image {self.count + 1}/{self.nf} {path}: ")
                self.count += 1  # move to the next file
                if self.count >= self.ni: # end of image list
                    break
        
        return paths, imgs, info
    
    def _new_video(self, path):
        """Create a new video capture object for the given path"""
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.cap.isOpened():
            raise FileNotFoundError(f'Failed to open video {path}')
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
    
    def __len__(self):
        """Returns the number of batches in the object"""
        return math.ceil(self.nf / self.bs)
    
class LoadPilAndNumpy:
    """
    Load images from PIL and Numpy arrays for batch processing
    
    This class is designed to manage loading and pre-processing of image data from both PIL and Numpy formats
    It performs basic validation and format conversion to ensure that the images are in the required format for downstream processing
    
    Attributes:
        paths (list): List of image paths or autogenerated filenames.
        im0 (list): List of images stored as Numpy arrays.
        mode (str): Type of data being processed, defaults to 'image'
        bs (int): Batch size, equivalent to the length of 'im0'
        
    Methods:
        _single_check(im): Validate and format a single image to a Numpy array
    """
    
    def __init__(self, im0):
        """Initialize PIL and Numpy Dataloader"""
        if not isinstance(im0, list):
            im0 = [im0]
        self.paths = [getattr(im, 'filename', f"image{i}.jpg") for i, im in enumerate(im0)]
        self.im0 = [self._single_check(im) for im in im0]
        self.mode = 'image'
        self.bs = len(self.im0)
        
    @staticmethod
    def _single_check(im):
        """Validate and format an image to numpy array"""
        assert isinstance(im, (Image.Image, np.ndarray)), f"Expected PIL/np.ndarray image type but got {type(im)}"
        if isinstance(im, Image.Image):
            if im.mode != "RGB":
                im = im.convert("RGB")
            im = np.asarray(im)[:,:,::-1]
            im = np.ascontiguousarray(im)
        return im
    
    def __len__(self):
        """returns the length of the 'im0' attribute"""
        return len(self.im0)
    
    def __next__(self):
        """Returns batch paths, images, processed images, None, ''."""
        if self.count == 1: # loop only once as it's batch inference
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, [""] * self.bs
    
    def __iter__(self):
        """Enables iteration for class LoadPILAndNumpy"""
        self.count = 0
        return self
    
class LoadTensor:  # TODO write LoadTensor
    pass
                
        
def autocast_list(source):   # TODO write autocast_list
    pass


# Define constants
LOADERS = (LoadStreams, LoadPilAndNumpy, LoadImagesAndVideos, LoadTensor)