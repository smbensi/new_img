# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L814

import contextlib

import torch
import torch.nn as nn

from img_xtend.utils import LOGGER, DEFAULT_CFG_DICT, emojis
from img_xtend.utils.checks import check_suffix

def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of model weights=[a, b, c] or a single model weights=[a] or weights=a"""
    
    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()
        
        # Model compatibility update
        model.args = args # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        # model.task = guess_model_task(model)
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])
        
        # append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()) # model in eval mode
        
    # Module updates
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
    
    # Return model
    if len(ensemble):
        return ensemble[-1]
    
    # Return ensemble
    LOGGER.info(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride 
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble

@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`)
    
    This function can be used to change the module paths during runtime. It's usefule when refactoring code, when you've moved a module from one location to another, but you still want to support the old import paths for backwards compatibility

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths. Defaults to None.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes. Defaults to None.
        
    Example:
        ```python
        with temporary_modules({'old.module': 'new.module'}, {'old.module.attribute': 'new.module.attribute'}):
            import old.module  # this will now import new.module
            from old.module import attribute  # this will now import new.module.attribute
        ```

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module
    
    try:
        # Set attributes in sys.modules under their old name
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]
    
    
def torch_safe_load(weight):
    """
    This function attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised,
    it catches the error, logs a warning message, and attempts to install the missing module via the
    check_requirements() function. After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.

    Returns:
        (dict): The loaded PyTorch model.
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
            },
        ):
            ckpt = torch.load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolov8n.pt'"
                )
            ) from e
        LOGGER.warning(
            f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolov8n.pt'"
        )
        # check_requirements(e.name)  # install missing module
        ckpt = torch.load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"WARNING ⚠️ The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file  # load

class Ensemble(nn.ModuleList):
    """Ensemble of models"""
    
    def __init__(self):
        """initialize an ensemble of models"""
        super().__init__()
        
    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLO network's final layer"""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 2)  # nms ensemble, y shape (B, HW, C)
        return y, None