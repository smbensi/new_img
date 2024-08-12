# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/model.py


import inspect
from pathlib import Path
from typing import List, Union

import numpy as np
import torch

class Model(nn.Module):
    """
    A base class for implementing YOLO models, unifying APIs across different model tpyes
    
    This class provides a common interface for various operations related to YOLO models, such as training, validation, exporting and benchmarking.
    It handles different types of models, including those loaded from local files,
    ultralytics HUB, or Triton server
    
    Attributes:
        callbacks (Dict): A dictionary of call back functions for various events during model operations
        predictor (BasePredictor): The predictor object used for making predictions.
        model (nn.Module): The underlying PyTorch model
        trainer (BaseTrainer): The trainer object used for training the model
        ckpt (Dict): The checkpoint data if the model is loaded from a *.pt file
        cfg (str): The configuration of the model if loaded from a *.yaml file
        ckpt_path (str): The path to the checkpoint file
        overrides (Dict): A dictionary of overrides for model configuration 
        metrics (Dict): the latest training/validation metrics
        session (HUBTrainingSession): The Ultraltics HUB session, if applicable
        task (str): The type of task the model is intended for
        model_name (str): the name of the model

    Methods:
        __call__ : Alias for the predict method, enabling the model instance to be callable
        _new : Initializes a new model based on a configuration file
        _load: Loads a model from a checkpoint file
        _check_is_pytorch_model: Ensures that the model is a PyTorch model
        reset_weights: Restes the model's weights to their initial state
        load : Loads model weights from a specified file
        save: Saves the current state of the model to a file
        info: logs or returns information about the model
        fuse: Fuses Conv2d and BatchNorm2d layers for optimized inference
        predict: preforms object detection predictions
        track: performs object tracking
        val: validates the model on a dataset
        benchmark: Benchmarks the model on various export formats
        export: Exports the model to different formats
        train: Trains the model on a dataset
        tune: Performs hyperparameter tuning
        _apply: Applies a function to the model's tensors
        add_callback : Adds a callback function for an event
        clear_callback : clears all callbacks function for an event
        reset_callback : Reset all callback to their default functions
        
    """
    
    def __init__(
        self,
        model: Union[str, Path] = "yolov8n.pt",
        task: str = None,
        verbose: bool = False
    ) -> None:
        """
        Initializes a new instance of the YOLO model class

        Args:
            model (Union[str, Path], optional): Path or name of the model to load or create. Can be a local file path, a model name from Ultralytics HUB, or a Triton Server model. Defaults to "yolov8n.pt".
            task (str | None): The task type associated with the YOLO model, specifying its application domain. Defaults to None.
            verbose (bool, optional): If True, enables verbose output during the model's initialization and subsequent operations. Defaults to False.
        """
        
        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None # reuse predictor
        self.model = None # model object
        self.trainer = None # trainer object
        self.ckpt = None # if loaded from *.pt 
        self.cfg = None # if loaded form *yaml
        self.ckpt_path = None
        self.overrides = {} # overrides for trainer object
        self.metrics = None # validation/training metrics
        self.session = None # HUB session
        self.task = task
        model = str(model).strip()
        
        # Check if Ultralytics HUB model from https://hub.ultralytics.com
        if self.is_hub_mode(model):
            # Fetch model from HUB
            checks.check_requirement("hub-sdk>=0.0.8")
            self.session = HUBTrainingSession.create_session(model)
            model = self.session.model_file
            
        # Check if triton server model
        elif self.is_triton_model(model):
            self.model_name = self.model = model
            return
        
        # Load or create new YOLO model
        if Path(model).suffix in {".yaml", ".yml"}:
            self._new(model, task=task, verbose=verbose)
        else:
            self._load(model, task=task)
            
    
    def __call__(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs,
    ) -> List:
        """
        Alias for the predict method enabling the model instance to be callable for predictions
        
        This method simplifies the process of making predicitons by allowing the model instance to be called directly with required arguments

        Args:
            source (Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor], optional): The source of the image(s) to make predicitions on. Defaults to None.
            stream (bool, optional): if True, treat the input source as a continuous stream for predicitions. Defaults to False.

        Returns:
            List[ultralytics.engine.results.Results]: A list of prediction resultsm each encapsulated in a Result object
        """
        return self.predict(source, stream, **kwargs)
            