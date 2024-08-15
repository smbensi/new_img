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

from img_xtend.utils import LOGGER, IS_JETSON, LINUX
from img_xtend.utils.downloads import is_url


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
            
        # Triton Server
        elif triton:
            check_requirements("tritonclient[all]")
            from img_xtend.utils.triton import TritonRemoteModel
            
            model = TritonRemoteModel(w)
        
        
        # TensorRT
        elif engine:
            LOGGER.info(f"Loading {w} for TensorRT inference...")
            try:
                import tensorrt as trt # noqa https://developer.nvidia.com/nvidia-tensorrt-download
            except ImportError:
                if LINUX:
                    check_requirements("tensorrt>7.0.0,<=10.1.0")
                import tensorrt as trt
            check_version(trt.__version__, ">=7.0.0", hard=True)
            check_version(trt.__version__, "<=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")
            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            # Read file
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                try:
                    meta_len = int.from_bytes(f.read(4), byteorder="little") # read metadata length
                    metadata = json.loads(f.read(meta_len).decode('utf-8')) # read metadata
                except UnicodeDecodeError:
                    f.seek(0) # engine file may lack embedded Ultralytics metadata
                model = runtime.deserialize_cuda_engine(f.read()) # read engine
                
            # model context
            try:
                context = model.create_execution_context()
            except Exception as e: # Model is None
                LOGGER.error(f"ERROR: TensorRT model exported with a different version than {trt.__version__}\n")
                raise e
            
            bindings = OrderedDict()
            output_names = []
            fp16 = False # default updated below
            dynamic = False
            is_trt10 = not hasattr(model, "num_bindings")
            num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
            for i in num:
                if is_trt10:
                    name = model.get_tensor_name(i)
                    dtype = trt.nptype(model.get_tensor_dtype(name))
                    is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                    if is_input:
                        if -1 in tuple(model.get_tensor_shape(name)):
                            dynamic = True
                            context.set_input_shape(name, tuple(model.get_tensor_profile_shape(name, 0)[1]))
                            if dtype == np.float16:
                                fp16 = True
                    else:
                        output_names.append(name)
                    shape = tuple(context.get_tensor_shape(name))
                else: # TensorRT < 10.0
                    name = model.get_binding_name(i)
                    dtype = trt.nptype(model.get_binding_dtype(i))
                    is_input = model.binding_is_input(i)
                    if model.binding_is_input(i):
                        if -1 in tuple(model.get_binding_shape(i)): # dynamic
                            dynamic = True
                            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[1]))
                        if dtype == np.float16:
                            fp16 = True
                        else:
                            output_names.append(name)
                    shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings["images"].shape[0] # if dynamic, this is instead max batch size
            
            
    
    def forward(self, im, augment=False, visualize=False, embed=None):
        pass
    
    def from_numpy(self, x):
        """
        Convert a numpy array to a tensor.

        Args:
            x (np.ndarray): The array to be converted.

        Returns:
            (torch.Tensor): The converted tensor
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x
    
    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        Warm up the model by running one forward pass with a dummy input

        Args:
            imgsz (tuple): The shape of the dummy input tensor in the format (batch_size, channels, height, width). Defaults to (1, 3, 640, 640).
        """
        import torchvision # (import here so torcvision import time not recorde in postprocess time)
        
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton, self.nn_module
        if any(warmup_types) and (self.device_type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device) # input
            for _ in range(2 if self.jit else 1):
                self.forward(im) # warmup
        
    
    @staticmethod
    def _model_type(p="path/to/model.pt"):
        """
        This function takes a path to a model file and returns the model type. Possibles types are pt, jit, onnx, xml,
        engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, ncnn or paddle.

        Args:
            p: path to the model file. Defaults to path/to/model.pt

        Examples:
            >>> model = AutoBackend(weights="path/to/model.onnx")
            >>> model_type = model._model_type()  # returns "onnx"
        """
        
        from img_xtend.engine.exporter import export_formats
        
        sf = list(export_formats().Suffix) # export suffixes
        if not is_url(p) and not isinstance(p, str):
            check_suffix(p, sf) # checks
        name = Path(p).name
        types = [s in name for s in sf]
        types[5] |= name.endswith(".mlmodel") # retain support for older Apple CoreML *.mlmodel formats
        types[8] &= not types[9] # tflite &= not edgetpu
        if any(types):
            triton = False
        else:
            from urllib.parse import urlsplit
            
            url = urlsplit(p)
            triton = bool(url.netloc) and bool(url.path) and url.scheme in {"http", "grpc"}
        
        return types + [triton]
                