import numpy as np
from pathlib import Path

from img_xtend.tracker.appearance.backends.base_backend import BaseModelBackend

from img_xtend.tracker.appearance.reid_model_factory import (
    load_pretrained_weights,
)


class PyTorchBackend(BaseModelBackend):

    def __init__(self, weights, device, half):
        super().__init__(weights, device, half)
        self.nhwc = False
        self.half = half

    def load_model(self, w):
        # Load a PyTorch model
        if w and w.is_file():
            load_pretrained_weights(self.model, w)
        self.model.to(self.device).eval()
        self.model.half() if self.half else self.model.float()

    def forward(self, im_batch):
        features = self.model(im_batch)
        return features
