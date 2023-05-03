from typing import Dict, Any
from PIL import Image
from time import time

import torch
import torch.nn.functional as F
import torchvision.transforms as T


def base_transformer(img: Image) -> torch.Tensor:
    tnsr = T.ToTensor()(img).unsqueeze(0)
    tnsr = F.interpolate(tnsr, size=(299, 299), mode="bilinear", align_corners=False)
    return tnsr.squeeze()


def clean_fid_transformer(img: Image) -> torch.Tensor:
    img = img.resize((299, 299), resample=Image.BICUBIC)
    tnsr = T.PILToTensor()(img)
    return tnsr.float() / 255


def torch_cov(m, rowvar=False):
    """
    Estimate a covariance matrix given data.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
    Returns:
        The covariance matrix of the variables.
    """
    if m.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


# Pytorch implementation of matrix sqrt, from Tsung-Yu Lin, and Subhransu Maji
# https://github.com/msubhransu/matrix-sqrt
def sqrt_newton_schulz(A, numIters, dtype=None):
    with torch.no_grad():
        if dtype is None:
            dtype = A.type()
        batchSize = A.shape[0]
        dim = A.shape[1]
        normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
        Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
        K = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1)
        Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1)
        K = K.type(dtype)
        Z = Z.type(dtype)
        for _ in range(numIters):
            T = 0.5 * (3.0 * K - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)
        sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA


class Registry:
    def __init__(self):
        self.items: Dict[str, Any] = {}

    def __getitem__(self, name: str) -> Any:
        item = self.items.get(name)

        if item is None:
            raise ValueError("Unknown metric or data type")

        return item

    def __setitem__(self, name: str, item: Any) -> None:
        if not isinstance(name, str):
            raise TypeError("Metric or data type should be str")

        self.items[name] = item

    def add_to_registry(self, key: str):

        def wraps(cls):
            self.items[key] = cls
            return cls

        return wraps


class Timer:
    def __init__(self, info=None, log_event=None):
        self.info = info
        self.log_event = log_event

    def __enter__(self):
        if torch.cuda.is_available():
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
        else:
            self.start = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            self.end.record()
            torch.cuda.synchronize()
            self.duration = self.start.elapsed_time(self.end) / 1000
        else: self.duration = time() - self.start
        if self.info is not None:
            self.info[f"duration/{self.log_event}"] = self.duration
