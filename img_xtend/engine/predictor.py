# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py#L186

import platform
import re
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

from img_xtend.utils import LOGGER
from img_xtend.data.build import load_inference_source


STREAM_WARNING = """
WARNING ⚠️ inference results will accumulate in RAM unless `stream=True` is passed, causing potential out-of-memory
errors for large sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.

Example:
    results = model(source=..., stream=True)  # generator of Results objects
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs
"""

class BasePredictor:
    """
    A bsae class for creating predictors
    
    Attributes:
        args (SimpleNamespace) : Configuration for the predictor
        save_dir (Path): Directory to save results
        done_warmup (bool): Whether the predictor has finished setup
        model (nn.Module): Model used for prediction
        data (dict): data configuration
        device (torch.device): device used for prediction
        dataset (Dataset): Dataset used for prediction
        vid_writer (dict): Dictionary of {save_path: video_writer, ... } writer for saving video output
    """
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None) -> None:
        """
        Initializes the BasePredictor class

        Args:
            cfg (str, optional): Path to configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): configuration overides. Defaults to None.
        """
        
        self.args = get_cfg(cfg, overrides) # TODO add get_cfg
        self.save_dir = get_save_dir(self.args)  # TODO add get_save_dir 
        if self.args.conf is None:
            self.args.conf = 0.25 # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)    # TODO add check_imshow
        
        # Usable if setup done
        self.model = None
        self.data = self.args.data
        self.imgz = None
        self.device = None
        self.dataset = None
        self.vid_writer = {}    # dict of {save_path: video_writer, ...}
        self.plotted_img = None
        self.source_type = None
        self.seen = 0
        self.windows = []
        self.batch = None
        self.batch = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks() # TODO add callbacks
        self.txt_path = None
        self._lock = threading.Lock()  # for automatic thread-safe inference
        callbacks.add_integration_callbacks(self)  # TODO add integration callbacks
        
        
    def preprocess(self, im):
        """
        Prepares input image before inference

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im)) 
            im = im[..., ::-1].transpose((0,3,1,2))  # BGR to RGB, BHWC to BCHW, (n,3,h,w)
            im = np.ascontiguousarray(im) # contiguous
            im = torch.from_numpy(im)
        
        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255   # 0 - 255.0 to 0.0 - 1.0
        return im
    
    def pre_transform(self, im):
        """
        Pre-transform input image before inference

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list
        
        Returns:
            (list): A list of transformed images
        """
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(self.imgz, auto=same_shapes and self.model.pt, stride=self.model.stride)  # TODO Letterbox
        return [letterbox(image=x) for x in im]
    
    def inference(self, im, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments"""
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and (not self.source_type.tensor)
            else False
        )
        return self.model(im, augment=self.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)
    
    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        """Performs inference on an image or stream"""
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(self.stream_inference(source, model, *args, **kwargs)) # merge list of Result into one
            
    
    @smart_inference_model() # TODO add smart inference model
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info("")
        
        # setup model
        if not self.model:
            self.setup_model(model) 
        
        with self._lock: # for thread-safe inference
            # setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)
            
            # check if save dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.args.save_dir).mkdir(parent=True, exist_ok=True)
            
            # warmup model
            if not self.done_warmup:
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True
                
            self.seen, self.windows, self.batch = 0, [], None
            
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            ) #TODO add profile
            self.run_callbacks("on_predict_start")
            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch
                
                # Preprocess
                with profilers[0]:
                    im = self.preprocess(im0s)
                
                # Inference
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds # yield embedding tensors
                        continue
                
                # Postprocess
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")
                
                # visualize, save, write results
                n = len(im0s)
                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess": profilers[0].dt + 1e3 / n,
                        "inference": profilers[1].dt + 1e3 / n,
                        "postprocess": profilers[2].dt + 1e3 / n,
                    }
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s[i] += self.write_results(i, Path(paths[i]), im, s)
                
                # Print batch results
                if self.args.verbose:
                    LOGGER.info("\n".join(s))
                
                self.run_callbacks("on_predict_batch_end")
                yield from self.results
            
            # Release assets
            for v in self.vid_writer.values():
                if isinstance(v, cv2.VideoWriter):
                    v.release()
                    
            # Print final results:
            if self.args.verbose and self.seen:
                t = tuple(x.t / self.seen * 1e3 for x in profilers) # speeds per image
                LOGGER.info(
                    f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                    f"{(min(self.args.batch, self.seen), 3, *im.shape[2:])}" % t
                )
            if self.args.save or self.args.save_txt or self.args.save_crop:
                nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
                s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
            self.run_callbacks("on_predict_end")

    
    def setup_model(self, model, verbose=True):
        """Initializes YOLO model with given parameters and set it to evaluation mode"""
        self.model = AutoBackend(
            weights=model or self.args.model,
            device=select_device(self.args.device, verbose=verbose),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            batch=self.args.batch,
            fuse=True,
            verbose=verbose,
        )   # TODO autobackend
        # TODO select device
        
        self.device = self.model.device # update device
        self.args.half = self.model.fp16 # update half
        self.model.eval()
    
    def setup_source(self, source):
        """Sets up source and inference mode"""
        self.imgz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2) # check image size 
        # TODO add check_imgsz
        self.transforms = (
            getattr(
                self.model.model,
                "transforms",
                classify_transforms(self.imgz[0], crop_fraction=self.args.crop_fraction),
            )
            if self.args.task == 'classify'
            else None
        )
        
        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer,
        )
        self.source_type = self.dataset.source_type
        if not getattr(self, "stream", True) and (self.source_type.stream
                                                  or self.source_type.screenshot
                                                  or len(self.dataset) >1000
                                                  or any(getattr(self.dataset, "video_flage", [False]))
                                                  ): # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_writer = {}
        