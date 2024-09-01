#  source : https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from img_xtend.utils import LOGGER

class BaseTensor(SimpleClass):
    """
    Base Tensor class with additional methods for easy manipulation and device handling
    
    Attributes:
        data (torch.Tensor | np.ndarray): Prediction data such as bounding boxes, masks, or keypoints.
        orig_shape (Tuple[int, int]): Original shape of the image, typically in the format (height, width)
    
    Methods:
        cpu: return a copy of the tensor stored in CPU memory
        numpy: Returns a copy of the tensor as a numpy array
        cuda: moves the tensor to GPU memory, returning a new instance if necessary
        to: return a copy of the tensor with the specified device and dtype
    """
    
    def __init__(self, data, orig_shape) -> None:
        """
        Initialize BaseTensor with prediction data and the original shape of the image

        Args:
            data (torch.Tensor | np.ndarray): Prediction data such as bounding boxes, masks, or keypoints
            orig_shape (Tuple[int, int]): Original shape of the image in (height, width) format.
        """
        assert isinstance(data, (torch.Tensor, np.ndarray)), "data muyst be torch.Tensor or np.ndarray"
        self.data = data
        self.orig_shape = orig_shape
    
    @property
    def shape(self):
        return self.data.shape

    def cpu(self):
        """
        Returns a copy of the tensor stored in CPU memory
        
        Returns:
            (BaseTensor): A new BaseTensor object with the data tensor moved to CPU memory
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)
    
    def numpy(self):
        """
        Returns a copy of the tensor as a numpy array
        
        Returns:
            (np.ndarray): A numpy array containing the same data as the original tensor
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)
    
    def cuda(self):
        """
        Moves the tensor to GPU memory
        
        Returns:
            (BaseTensor): A new BaseTensor instance with the data moved to GPU memory if it's
            not already a numpy arraym otherwise return self
        """
        return self.__class__(torch.astensor(self.data).cuda(), self.orig_shape)
    
    def to(self, *args, **kwargs):
        """
        Return a copy of the tensor with the specified device and dtype
        
        Args:
            *args (Any): Variable length argument list to be passed to torch.Tensor.to().
            **kwargs (Any): arbitrary keyword arguments to be passed to torch.Tensor.to().
        
        Returns:
            (BaseTensor): A new BaseTensor instance with the data moved to the specified device and/or dtype
        """
        return self.__class__(torch.astensor(self.data).to(*args,**kwargs), self.orig_shape)
    
    def __len__(self):
        """
        Returns the length of the underlying data tensor
        
        Returns:
            (int): The number of elements in the 1st dimension of the data tensor
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns a new BaseTensor instance containing the specified indexed elements of the data tensor

        Args:
            idx (int | List[int] | torch.Tensor): Index or indices to select from the data tensor
            
        Returns:
            (BaseTensor): A new BaseTensor instance containing the indexed data
        """
        return self.__class__(self.data[idx], self.orig_shape)
    

class Results(SimpleClass):
    pass