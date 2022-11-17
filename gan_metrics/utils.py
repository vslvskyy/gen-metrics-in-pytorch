import functools
import os
from glob import glob
from typing import List, Optional, Callable
from PIL import Image

import torch
import torch.nn.functional as F

from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from torch.utils.data import Dataset


def base_transformer(
        img: Image
    ):
    tnsr = T.ToTensor()(img).unsqueeze(0)
    tnsr = F.interpolate(
        tnsr,
        size=(299, 299),
        mode='bilinear',
        align_corners=False
    )
    return tnsr.squeeze()


def clean_fid_transformer(
        img: Image
    ):
    img = img.resize((299, 299), resample=Image.BICUBIC)
    img = T.PILToTensor()(img)
    return img.float() / 255


class Registry():
    def __init__(self):
        self.registry_dct = {
            "metrics": {},
            "data_types": {}
        }

    def fill_dct(self, key: str, is_metric: bool = False):

        def wraps(func):

            if is_metric:
                self.registry_dct["metrics"][key] = func
            else:
                self.registry_dct["data_types"][key] = func

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return wraps

reg_obj = Registry()


@reg_obj.fill_dct("StandardDataset")
class MyCifarDataset(CIFAR10):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable],
        train: bool = True,
        download: bool = True,
        **kwargs
    ):
        super().__init__(
            root="./",
            train=train,
            transform=transform,
            download=download
        )

    def __getitem__(self, idx):
        return super().__getitem__(idx)[0]


@reg_obj.fill_dct("ImageDataset")
class ImageDataset(Dataset):
    """A simple image dataset for calculating inception score and FID."""

    def __init__(
        self,
        root: str,
        transform  = None,
        num_images = None,
        exts: List[str] = ['png', 'jpg', 'JPEG'],
        **kwargs
    ):
        """Construct an image dataset.
        Args:
            root: Path to the image directory. This directory will be
                  recursively searched.
            exts: List of extensions to search for.
            transform: A torchvision transform to apply to the images. If
                       None, the images will be converted to tensors.
            num_images: The number of images to load. If None, all images
                        will be loaded.
        """
        self.paths = []

        for ext in exts:
            self.paths.extend(
                list(glob(
                    os.path.join(root, '*.%s' % ext), recursive=True)))

        self.paths = self.paths[:num_images]
        self.images = [Image.open(path).convert("RGB") for path in self.paths]

        self.transform = transform if transform is not None else T.ToTensor()

    def __len__(self):              # noqa
        return len(self.paths)

    def __getitem__(self, idx):     # noqa
        return self.transform(self.images[idx])


@reg_obj.fill_dct("stats")
def stats_type(
    root: str,
    **kwargs
):
    return root


class Timer:
    def __init__(self, info=None, log_event=None):
        self.info = info
        self.log_event = log_event

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end.record()
        torch.cuda.synchronize()
        self.duration = self.start.elapsed_time(self.end) / 1000
        if self.info is not None:
            self.info[f"duration/{self.log_event}"] = self.duration
