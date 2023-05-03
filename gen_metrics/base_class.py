from typing import Optional

import torch

from .utils import Registry, base_transformer


metrics = Registry()


class BaseMetric(object):
    """
    Base class for Generative Model's metric calculation
    """
    def __init__(
        self,
        device: torch.device = torch.device("cuda:0"),
        **kwargs
    ):
        self.device = device
        self.inception_model = None
        self.transform = base_transformer

    def __call__(
        self, gen_path: str, real_path: Optional[str] = None, gen_type: str = "folder", real_type: str = "folder"
    ):
        raise NotImplementedError("__call__ is not implemented")
