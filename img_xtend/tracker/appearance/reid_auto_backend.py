import torch
from pathlib import Path
from typing import Union, Tuple

from img_xtend.tracker.tracker_settings import WEIGHTS
from img_xtend.utils import LOGGER
from img_xtend.tracker.utils.torch_utils import select_device
# TODO
from img_xtend.tracker.appearance import export_formats
from img_xtend.tracker.appearance.backends.base_backend import BaseModelBackend
from img_xtend.tracker.appearance.backends.pytorch_backend import PyTorchBackend
# from boxmot.appearance.backends.onnx_backend import ONNXBackend
# from boxmot.appearance.backends.openvino_backend import OpenVinoBackend
# from boxmot.appearance.backends.tensorrt_backend import TensorRTBackend
# from boxmot.appearance.backends.tflite_backend import TFLiteBackend
# from boxmot.appearance.backends.torchscript_backend import TorchscriptBackend
# END TODO



class ReidAutoBackend():
    def __init__(
        self,
        weights: Path = WEIGHTS / "osnet_x1_0.pt",
        device: torch.device = torch.device("cpu"),
        half: bool = False) -> None:
        """
        Initializes the ReidAutoBackend instance with specified weights, device, and precision mode.

        Args:
            weights (Union[str, List[str]]): Path to the model weights. Can be a string or a list of strings; if a list, the first element is used.
            device (torch.device): The device to run the model on, e.g., CPU or GPU.
            half (bool): Whether to use half precision for model inference.
        """
        super().__init__()
        w = weights[0] if isinstance(weights, list) else weights
        (
            self.pt,
            self.jit,
            self.onnx,
            self.xml,
            self.engine,
            self.tflite,
        ) = self.model_type(w)  # get backend

        self.weights = weights
        self.device = select_device(device)
        self.half = half


    def get_backend(self): # -> Union['PyTorchBackend', 'TorchscriptBackend', 'ONNXBackend', 'TensorRTBackend', 'OpenVinoBackend', 'TFLiteBackend']:
        """
        Returns an instance of the appropriate backend based on the model type.

        Returns:
            An instance of a backend class corresponding to the detected model type.
        
        Raises:
            SystemExit: If no supported model framework is detected.
        """

        # Mapping of conditions to backend constructors
        backend_map = {
            self.pt: PyTorchBackend,
            # self.jit: TorchscriptBackend,
            # self.onnx: ONNXBackend,
            # self.engine: TensorRTBackend,
            # self.xml: OpenVinoBackend,
            # self.tflite: TFLiteBackend
        }

        # Iterate through the mapping and return the first matching backend
        for condition, backend_class in backend_map.items():
            if condition:
                return backend_class(self.weights, self.device, self.half)

        # If no condition is met, log an error and exit
        LOGGER.error("This model framework is not supported yet!")
        exit()


    def forward(self, im_batch: torch.Tensor) -> torch.Tensor:
        """
        Processes an image batch through the selected backend and returns the processed batch.

        Args:
            im_batch (torch.Tensor): The batch of images to process.

        Returns:
            torch.Tensor: The processed image batch.
        """
        im_batch = self.backend.preprocess_input(im_batch)
        return self.backend.get_features(im_batch)


    def check_suffix(self, file: Path = "osnet_x0_25_msmt17.pt", suffix: Union[str, Tuple[str, ...]] = (".pt",), msg: str = "") -> None:
        """
        Validates that the file or files have an acceptable suffix.

        Args:
            file (Union[str, List[str], Path]): The file or files to check.
            suffix (Union[str, Tuple[str, ...]]): Acceptable suffix or suffixes.
            msg (str): Additional message to log in case of an error.
        """

        suffix = [suffix] if isinstance(suffix, str) else list(suffix)
        files = [file] if isinstance(file, (str, Path)) else list(file)

        for f in files:
            file_suffix = Path(f).suffix.lower()
            if file_suffix and file_suffix not in suffix:
                LOGGER.error(f"File {f} does not have an acceptable suffix. Expected: {suffix}")


    def model_type(self, p: Path) -> Tuple[bool, ...]:
        """
        Determines the model type based on the file's suffix.

        Args:
            path (str): The file path to the model.

        Returns:
            Tuple[bool, ...]: A tuple of booleans indicating the model type, corresponding to pt, jit, onnx, xml, engine, and tflite.
        """

        sf = list(export_formats().Suffix)  # export suffixes
        self.check_suffix(p, sf)  # checks
        types = [s in Path(p).name for s in sf]
        return types