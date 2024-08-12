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
        f
    