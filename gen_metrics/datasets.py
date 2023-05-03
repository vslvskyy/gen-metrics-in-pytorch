import os
import torchvision.transforms as T

from glob import glob
from typing import List, Optional, Callable
from PIL import Image

from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset

from .utils import Registry


datasets = Registry()


@datasets.add_to_registry("folder")
class ImageDataset(Dataset):
    """A simple image dataset for calculating Generative Models' Metrics."""

    def __init__(
        self,
        root: str,
        transform=None,
        num_images=None,
        exts: List[str] = ["png", "jpg", "JPEG"],
        **kwargs
    ):
        """Construct an image dataset.
        Args:
            root: Path to the image directory. This directory will be recursively searched.
            transform: transform to apply to the images. If None, the images will be converted to tensors.
            num_images: The number of images to load. If None, all images will be loaded.
            exts: List of extensions to search for.
        """
        self.paths = []

        for ext in exts:
            self.paths.extend(
                list(glob(os.path.join(root, "*.%s" % ext), recursive=True))
            )

        self.paths = self.paths[:num_images]
        self.images = [Image.open(path).convert("RGB") for path in self.paths]

        self.transform = transform if transform is not None else T.ToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Image:
        return self.transform(self.images[idx])


@datasets.add_to_registry("cifar10")
class NoneLabelCifar10(CIFAR10):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable],
        train: bool = True,
        download: bool = True,
        **kwargs
    ):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )

    def __getitem__(self, idx: int) -> Image:
        return super().__getitem__(idx)[0]
