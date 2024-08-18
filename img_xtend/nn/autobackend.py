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

from img_xtend.utils import LOGGER, IS_JETSON, LINUX, ROOT, yaml_load
from img_xtend.utils.downloads import is_url
from .tasks import attempt_load_weights

def check_class_names(names):
    """
    Check class names
    
    Map imagenet class codes to human readable names if required. Convert lists to dicts
    """
    if isinstance(names, list):    #  names is a list
        names = dict(enumerate(names))     # convert to dict
    if isinstance(names, dict):
        # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'
        names = {int(k): str(v) for k,v in names.items()}
        n = len(names)
        if max(names.keys())>= n:
            raise KeyError(
                f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices "
                f"{min(names.keys())}-{max(names.keys())} defined in your dataset YAML."
            )
            
        if isinstance(names[0], str) and names[0].startswith("n0"):      # imagenet class codes, i.e. 'n01440764'
            names_map = yaml_load(ROOT / "cfg/datasets/ImageNet.yaml")["map"] # human-readable names
            names = {k: names_map[v] for k,v in names.items()}
    return names

def default_class_names(data=None):
    """Applies default class names to an input YAML file or returns numerical class names."""
    if data:
        with contextlib.suppress(Exception):
            return yaml_load(check_yaml(data)["names"])
    return {i: f"class{i}" for i in range(999)} # return default if above errors
    


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
        
        # Any other format (unsupported)
        else:
            from img_xtend.engine.exporter import export_formats
            
            raise TypeError(
                f"model='{w} is not a supported model format. "
                f"See https://docs.ultralytics.com/modes/predict for help.\n\n{export_formats()}"
            )
            
        # Load external metadata YAML
        if isinstance(metadata, (str, Path)) and Path(metadata).exists():
            metadata = yaml_load(metadata)
        if metadata and isinstance(metadata, dict):
            for k, v in metadata.items():
                if k in {"stride", "batch"}:
                    metadata[k] = int(v)
                elif k in {"imgsz", "names", "kpt_shape"} and isinstance(v, str):
                    metadata[k] = eval(v)
            stride = metadata["stride"]
            task = metadata["task"]
            batch = metadata["batch"]
            imgsz = metadata["imgsz"]
            names = metadata["names"]
            kpt_shape = metadata.get("kpt_shape")
        elif not (pt or triton or nn_module):
            LOGGER.warning(f"WARNING Metadata not found for model={weights}")
        
        # Check names
        if "names" not in locals(): # names missing
            names = default_class_names(data)
        names = check_class_names(names)
        
        # Disable gradients
        if pt:
            for p in model.parameters():
                p.requires_grad = False
        
        self.__dict__.update(locals())   # assign all variables to self
            
            
    
    def forward(self, im, augment=False, visualize=False, embed=None):
        """
        Runs inference on the YOLOv8 Multibackend model
        
        Args:
            im (torch.Tensor): The image tensor to perform inference on
            augment (bool): whether to perform data augmentation during inference, defaults to False
            visualize (bool): Whether to visualize the output predictions, default to False
            embed (list, optional): A list of feature vectors/ embeddings to return
            
        Returns:
            (tuple): Tuple containing the raw output tensor, and processed output for visualization (if visualize=True)
        """
        b, ch, h, w = im.shape # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0,2,3,1)  # torch BCHW ti numpy BHWC shape
            
        # PyTorch
        if self.pt or self.nn_module:
            y = self.model(im, augment=augment, visualize=visualize, embed=embed)
        
        # TorchScript
        elif self.jit:
            y = self.model(im)
            
        # ONNX OpenCV DNN
        elif self.dnn:
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
            
        # ONNX Runtime
        elif self.onnx:
            im = im.cpu().numpy()
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
            
            
        # TensorRT
        elif self.engine:
            if self.dynamic or im.shape != self.bindings["images"].shape:
                if self.is_trt10:
                    self.context.set_input_shape("images", im.shape)
                    self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                    for name in self.output_names:
                        self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(name)))
                else:
                    i = self.model.get_binding_index("images")
                    self.context.set_binding_shape(i, im.shape)
                    self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                    for name in self.output_names:
                        i = self.model.get_binding_index(name)
                        self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            
            s = self.bindings["images"].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs["images"] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        
        
        # Triton Server
        elif self.triton:
            im = im.cpu().numpy()
            y = self.model(im)
    
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
                