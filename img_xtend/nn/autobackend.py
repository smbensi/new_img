# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/autobackend.py

import ast
import contextlib
import json
import platform
import zipfile
from collections import OrderedDict, namedtuple
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from img_xtend.utils import LOGGER, IS_JETSON

class AutoBackend(nn.Module):
    """
    Handles dynamic backend selection for running inference using Ultralytics YOLO models
    
    The AutoBackend class is designed to provide an abstraction layer for various inference enfines.
    It supports a wide range of formats, each with specific naming conventions as outlined below:

        Supported Formats and Naming Conventions:
            | Format                | File Suffix      |
            |-----------------------|------------------|
            | PyTorch               | *.pt             |
            | TorchScript           | *.torchscript    |
            | ONNX Runtime          | *.onnx           |
            | ONNX OpenCV DNN       | *.onnx (dnn=True)|
            | OpenVINO              | *openvino_model/ |
            | CoreML                | *.mlpackage      |
            | TensorRT              | *.engine         |
            | TensorFlow SavedModel | *_saved_model    |
            | TensorFlow GraphDef   | *.pb             |
            | TensorFlow Lite       | *.tflite         |
            | TensorFlow Edge TPU   | *_edgetpu.tflite |
            | PaddlePaddle          | *_paddle_model   |
            | NCNN                  | *_ncnn_model     |

    This class offers dynamic backend switching capabilities based on the input model format, making it easier to deploy models across various platforms
    """
    
    @torch.no_grad()
    def __init__(
        self,
        weights="yolov8n.pt",
        device=torch.device("cpu"),
        dnn=False,
        data=None,
        fp16=False,
        batch=1,
        fuse=True,
        verbose=True,
    ):
        """
        Initialize the AutoBackend for inference
        
        Args:
            weights (str): Path to the model weights file. Defaults to 'yolov8n.pt'.
            device (torch.device): Device to run the model on. Defaults to CPU
            dnn (bool): use OpenCV DNN module for ONNX inference. Defaults to False
            data (str | Path | Optional): Path to the additional data.yaml file containing class names. Optional
            fp16 (bool): Enable half-precision inference. Supported only on specific backends. Default to False
            batch (int): Batch-size to assume for inference
            fuse (bool): Fuse Conv2D + BatchNorm layers for optimization. Defaults to True
            verbose (bool): Enable verbose logging. Defaults to True
        """
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        nn_module = isinstance(weights, torch.nn.Module)
        (
            pt,
            jit,
            onnx,
            xml,
            engine,
            coreml,
            saved_model,
            pb,
            tflite,
            edgetpu,
            tfjs,
            paddle,
            ncnn,
            triton,
        ) = self._model_type(w)
        
        fp16 &= pt or jit or onnx or xml or engine or nn_module or triton # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu # BHWC formats (vs torch BCHW)
        stride = 32 # default stride
        model, metadata = None, None
        
        # Set device
        cuda = torch.cuda.is_available() and device.type != "cpu" # use CUDA
        if cuda and not any([nn_module, pt, jit, engine, onnx]): # GPU dataloader formats
            device = torch.device("cpu")
            cuda = False
            
        # Download if not local
        if not (pt or triton or nn_module):
            w = attempt_download_asset(w) # TODO add attempt_download_asset
        
        # In-memory PyToch model
        if nn_module:
            model = weights.to(device)
            if fuse:
                model = model.fuse(verbose=verbose)
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape # pose-only
            stride = max(int(model.stride.max()), 32) # model stride
            names = model.module.names if hasattr(model, "module") else model.names
            model.half() if fp16 else model.float()
            self.model = model # explicitly assign for to(), cpu(), cuda(), half()
            pt = True
            
        # PyTorch
        elif pt:
            model = attempt_load_weights(
                weights if isinstance(weights,list) else w,
                device=device, inplace=True, fuse=fuse
            ) # TODO add attempt load weights
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape # pose-only
            stride = max(int(model.stride.max()), 32) # model stride
            names = model.module.names if hasattr(model, "module") else model.names # get class names
            model.half() if fp16 else model.float()
            self.model = model
            
        # TorchScript
        elif jit:
            LOGGER.info(f"Loading {w} for TorchScript inference")
            extra_files = {"config.txt": ""} # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files["config.txt"]:  # load metadata dict
                metadata = json.loads(extra_files["config.txt"], object_hook=lambda x: dict(x.items()))
        
        # ONNX runtime
        elif onnx:
            LOGGER.info(f"Loading {w} for ONNX Runtime inference...")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime")) # TODO add check_requirements
            if IS_JETSON:
                # Fix 'numpy.linalg._umath_linalg' has no attribute '_ilp64' for TF SavedModel on RPi and Jetson
                check_requirements("numpy==1.23.5")
            import onnxruntime
            
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            metadata = session.get_modelmeta().custom_metadata_map
            