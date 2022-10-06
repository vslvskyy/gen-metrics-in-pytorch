import torch
import numpy as np

from torchvision.datasets import CIFAR10
from PIL import Image
import torchvision.transforms as transforms

import os
from typing import List, Union, Tuple, Optional
from glob import glob

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


def resize_single_channel(
        x_np: np.ndarray
    ):
    img = Image.fromarray(x_np.astype(np.float32), mode='F')
    img = img.resize((299, 299), resample=Image.BICUBIC)
    return np.asarray(img).clip(0, 255).reshape(299, 299, 1)


def clean_fid_transformer_original(
        img: Image
    ):

    x = [resize_single_channel(np.asarray(img)[:, :, idx]) for idx in range(3)]
    x = np.concatenate(x, axis=2).astype(np.float32)
    return torch.transpose(torch.from_numpy(x), 0, -1)/255


def clean_fid_transformer(
        img: Image
    ):
    img = img.resize((299,299), resample=Image.BICUBIC)
    img = transforms.PILToTensor()(img)
    return img.float()/255


class MyCifarDataset(CIFAR10):
    def __getitem__(self, idx):
        return super().__getitem__(idx)[0]


def make_dataset(data: str, root: str= None, exts: str = None):
    if data.lower() == "cifar10":
        return MyCifarDataset(CIFAR10)
    return ImageDataset(root, exts)


class ImageDataset(Dataset):
    """A simple image dataset for calculating inception score and FID."""

    def __init__(self, root, exts=['png', 'jpg', 'JPEG'], transform=None,
                 num_images=None):
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
        self.transform = transform
        for ext in exts:
            self.paths.extend(
                list(glob(
                    os.path.join(root, f'*.%s' % ext), recursive=True)))
        self.paths = self.paths[:num_images]

    def __len__(self):              # noqa
        return len(self.paths)

    def __getitem__(self, idx):     # noqa
        image = Image.open(self.paths[idx])
        image = image.convert('RGB')        # fix ImageNet grayscale images
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = to_tensor(image)
        return image


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
