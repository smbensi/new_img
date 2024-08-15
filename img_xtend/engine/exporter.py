# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/exporter.py#L147

"""
Export a YOLOv8 Pytorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit

Format                 | `format=argument`              | Model
---                    | ---                            | ---
PyTorch                | -                              | yolov8n.pt
TorchScript            | `torchscript`                  | yolov8n.torchscript
ONNX                   | `onnx`                         | yolov8n.onnx
OpenVINO               | `openvino`                     | yolov8n_openvino_model/
TensorRT               | `engine`                       | yolov8n.engine
CoreML                 | `coreml`                       | yolov8n.mlpackage
TensorFlow SavedModel  | `saved_model`                  | yolov8n_saved_model/
TensorFlow GraphDef    | `pb`                           | yolov8n.pb
TensorFlow Lite        | `tflite`                       | yolov8n.tflite
TensorFlow Edge TPU    | `edgetpu`                      | yolov8n_edgetpu.tflite
TensorFlow.js          | `tfjs`                         | yolov8n_web_model/
PaddlePaddle           | `paddle`                       | yolov8n_paddle_model/
NCNN                   | `ncnn`                         | yolov8n_ncnn_model/


"""
from pathlib import Path
import gc
import json

import torch


from img_xtend.utils import (
    LOGGER,
    LINUX,
    DEFAULT_CFG,
    get_default_args,
    colorstr
)
from img_xtend.utils.ops import Profile
from img_xtend.utils.files import file_size

def export_formats():
    """YOLOv8 export formats"""
    import pandas # scope for faster 'import ultralytics'
    
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["CoreML", "coreml", ".mlpackage", True, False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", True, False],
        ["TensorFlow.js", "tfjs", "_web_model", True, False],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True],
        ["NCNN", "ncnn", "_ncnn_model", True, True],
    ]
    return pandas.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])

def try_export(inner_func):
    """YOLOv8 export decorator, i.e, @try_export"""
    inner_args = get_default_args(inner_func)
    
    def outer_func(*args, **kwargs):
        """Export a model"""
        prefix = inner_args["prefix"]
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f"{prefix} export success {dt.t:.1f}s, saved as '{f}' ({file_size(f):.1f} MB)")
            return f, model
        except Exception as e:
            LOGGER.info(f'{prefix} export failure ❌ {dt.t:.1f}s: {e}')
            raise e
    
    return outer_func
                
class Exporter:
    """
    A class for exporting a model
    
    Attributes:
        args (SimpleNamespace): Configuration for the exporter
        callbacks (list, optional): List of callback functions. Defaults to None
    """
    
    def __init__(self, cfg=DEFAULT_CFG, overrrides=None, _callbacks=None):
        """
        Initializes the Exporter class

        Args:
            cfg (_type_, optional): _description_. Defaults to DEFAULT_CFG.
            overrrides (_type_, optional): _description_. Defaults to None.
            _callbacks (_type_, optional): _description_. Defaults to None.
        """
    
    @try_export
    def export_engine(self, prefix=colorstr("TensorRT:")):
        """YOLOv8 TensorRT export https://developer.nvidia.com/tensorrt."""
        assert self.im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. use 'device=0'"
        f_onnx, _ = self.export_onnx() # run before TRT import https://github.com/ultralytics/ultralytics/issues/7016
        
        try:
            import tensorrt as trt
        except ImportError:
            if LINUX:
                check_requirements("tensorrt>7.0.0,<=10.1.0")
            import tensorrt as trt
        check_version(trt.__version__, ">=7.0.0", hard=True)
        check_version(trt.__version__, "<=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")
        
        # Setup and checks
        LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
        is_trt10 = int(trt.__version__.split(".")[0]) >= 10 
        assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"
        f = self.file.with_suffix(".engine")  # TensorRT engine file
        logger = trt.Logger(trt.Logger.INFO)
        if self.args.verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE
        
        # Engine builder
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        workspace = int(self.args.workspace * (1 << 30))
        if is_trt10:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
        else:
            config.max_workspace_size = workspace
        flag = 1 << int(trt.NetworkDefinitionCreationFlage.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        half = builder.platform_has_fast_fp16 and self.args.half
        int8 = builder.platform_has_fast_int8 and self.args.int8
        # Read ONNX file
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(f_onnx):
            raise RuntimeError(f"failed to load ONNX file: {f_onnx}")
        
        # Network inputs
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')
            
        if self.args.dynamic:
            shape = self.im.shape
            if shape[0] <= 1:
                LOGGER.warning(f"{prefix} WARNING 'dynamic=True' model requires max batch size, i.e. 'batch=16'")
            profile = builder.create_optimization_profile()
            min_shape = (1, shape[1], 32, 32)  # minimum input shape
            max_shape = (*shape[:2],  *(max(1, self.args.workspace) * d for d in shape[2:]))    # max input shape
            for inp in inputs:
                profile.set_shape(inp.name, min=min_shape, opt=shape, max=max_shape)
            config.add_optimization_profile(profile)
        
        LOGGER.info(f"{prefix} building {'INT8' if int8 else 'FP' + ('16' if half else '32')} engine as {f}")
        if int8:
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_calibration_profile(profile)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            
            class EngineCalibrator(trt.IInt8Calibrator):
                def __init__(
                    self,
                    dataset, # ultralytics.data.build.InfiniteDataLoader
                    batch: int,
                    cache: str = "",
                ) -> None:
                    trt.IInt8Calibrator.__init__(self)
                    self.dataset = dataset
                    self.data_iter = iter(dataset)
                    self.algo = trt.Calibration.ENTROPY_CALIBRATION_2
                    self.batch = batch
                    self.cache = Path(cache)
                    
                def get_algorithm(self) -> trt.CalibrationAlgoType: 
                    """Get the calibration algorithm to use"""
                    return self.algo
                
                def get_batch_size(self) -> int:
                    """Get the batch size to use for calibration"""
                    return self.batch or 1
                
                def get_batch(self, names) -> list:
                    """Get the next batch to use for calibration, as a list of device memory pointers"""
                    try:
                        im0s = next(self.data_iter)["img"] / 255.0
                        im0s = im0s.to("cuda") if im0s.device_type == "cpu" else im0s
                        return [int(im0s.data_ptr())]
                    except StopIteration:
                        # Return [] or None, signal to TensorRT there is no calibration data remaining
                        return None
                    
                def read_calibration_cache(self) -> bytes:
                    """Use existing cache instead of calibrating again, otherwise, implicitly return None"""
                    if self.cache.exists() and self.cache.suffix == '.cache':
                        return self.cache.read_bytes()
                    
                def write_calibration_cache(self, cache) -> None:
                    """Write calibration cache to disk"""
                    _ = self.cache.write_bytes(cache)
            
            # Load dataset w/ builder (for batching) and calibrate
            config.int8_calibrator = EngineCalibrator(
                dataset=self.get_int8_calibration_dataloader(prefix),
                batch = 2 * self.args.batch,
                cache = str(self.file.with_suffix(".cache")),
            )
        
        elif half:
            config.set_flag(trt.BuilderFlag.FP16)
            
        # Free CUDA memory
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        
        # Write file
        build = builder.build_serialized_network if is_trt10 else builder.build_engine
        with build(network, config) as engine, open(f, "wb") as t:
            # Metadata
            meta = json.dumps(self.metadata)
            t.write(len(meta).to_bytes(4, byteorder='little', signed=True))
            t.write(meta.encode())
            # Model
            t.write(engine if is_trt10 else engine.serialize())
        
        return f, None

        
        
    