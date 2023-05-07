from typing import Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader

from .base_class import BaseMetric, metrics
from .datasets import datasets
from .inception_net import InceptionV3


@metrics.add_to_registry("is")
class InceptionScore(BaseMetric):
    """
    Class for Inception Score calculation
    https://arxiv.org/pdf/1606.03498.pdf
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
        splits_n: int = 1,
        **kwargs
    ):
        """
        Args:
            device: what torch device to use for calculations
            splits_n: number of batchs to calculate IS for each one
        """
        super().__init__(device)
        self.inception_model = InceptionV3(
            [InceptionV3.BLOCK_INDEX_BY_DIM[1008]], resize_input=False
        )
        self.splits_n = splits_n

    @staticmethod
    def compute_is_from_probs(probs: torch.Tensor) -> float:
        """
        Batch Inception Score calculation

        Args:
            probs: inception features (conditional probabilities of classes)
        returns:
            inseption score value
        """

        kl = probs * (
            torch.log(probs) - torch.log(torch.unsqueeze(torch.mean(probs, 0), 0))
        )
        kl = torch.mean(torch.sum(kl, 1))
        res = torch.exp(kl)

        return res

    def compute_probs(self, path: str, data_type: str, save_path=None, **kwargs) -> torch.Tensor:
        """
        Compute classes probabilities of given data from Inception Model

        Args:
            path: path to images or precalculated probabilities
            data_type: folder, stats or some standard dataset (e. g. cifar10)
            save_path: where to save probabilities
        returns:
            ftrs: classes probabilities from Inception Model
        """
        if data_type == "stats":
            probs = torch.tensor(np.load(path)).to(self.device)
        else:
            data = datasets[data_type](
                root=path,
                transform=self.transform,
                train=True,
                download=True,
                **kwargs
            )
            data = DataLoader(data, batch_size=50, num_workers=2)
            probs = self.inception_model.get_features(data, dim=1008, device=self.device)

            if save_path is not None:
                np.save(save_path, probs.cpu().numpy())

        return probs

    def __call__(self, gen_path, gen_type="folder", gen_save_path=None, **kwargs) -> Tuple[float, float]:
        """
        Calculates Inception Score of given data

        Args:
            gen_path: path to generated samples or features from Inception Model
            gen_type: type of generated data (folder, stats, or standard dataset)
            gen_save_path: path where to save generated Inception Model features
        returns:
            mean: mean of data parts' inception scores
            std: std of data parts' inception scores
        """
        probs = self.compute_probs(gen_path, gen_type, gen_save_path, **kwargs)
        scores = []
        for probs_batch in torch.split(probs, split_size_or_sections=probs.shape[0] // self.splits_n):
            score = self.compute_is_from_probs(probs_batch)
            scores.append(score)

        scores = torch.stack(scores)
        mean_score = torch.mean(scores).cpu().item()
        std = torch.std(scores).cpu().item()

        return (mean_score, std)
