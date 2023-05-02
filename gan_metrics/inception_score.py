from typing import Tuple, Union

import torch
import numpy as np
from torch.utils.data import DataLoader

from .base_class import BaseGanMetric, metrics
from .inception_net import InceptionV3
from .utils import Timer, datasets, base_transformer


@metrics.add_to_registry("is")
class InceptionScore(BaseGanMetric):
    """
    Class for Inception Score calculation
    """

    def __init__(self, device=torch.device("cuda:0"), splits_n: int = 10, **kwargs):
        """
        Args:
            splits_n: number of batchs to calculate IS for each one
            device: what model device to use
        """

        self.device = device
        self.splits_n = splits_n
        self.inception_model = InceptionV3(
            [InceptionV3.BLOCK_INDEX_BY_DIM[1008]], resize_input=False
        )
        self.transform = base_transformer

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

    def compute_probs(self, path: str, data_type: str, save_path=None) -> torch.Tensor:
        if data_type == "stats":
            probs = torch.tensor(np.load(path)).to(self.device)
        else:
            data = datasets[data_type](
                root=path,
                transform=self.transform,
                train=True,
                download=True
            )
            data = DataLoader(data, batch_size=50, num_workers=2)
            probs = self.inception_model.get_features(data, dim=1008, device=self.device)

            if save_path is not None:
                np.save(save_path, probs.cpu().numpy())

        return probs

    def __call__(self, gen_path, gen_type="folder", gen_save_path=None, **kwargs) -> Tuple[float, float]:
        """
        Inception Score calculation

        Args:
            train: generated data
            probs: inception features (conditional probabilities of classes)
        returns:
            mean and standard deviation of batchs' IS
        """
        probs = self.compute_probs(gen_path, gen_type, gen_save_path)
        scores = []
        for _, probs_batch in enumerate(
            torch.split(
                probs, split_size_or_sections=probs.shape[0] // self.splits_n
            ),
            start=0,
        ):
            score = self.compute_is_from_probs(probs_batch)
            scores.append(score)

        scores = torch.stack(scores)
        mean_score = torch.mean(scores).cpu().item()
        std = torch.std(scores).cpu().item()

        return (mean_score, std)
