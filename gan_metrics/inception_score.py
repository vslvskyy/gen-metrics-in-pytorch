from typing import Tuple, Union

import torch
import numpy as np
from torch.utils.data import DataLoader

from .base_class import BaseGanMetric, metrics
from .inception_net import InceptionV3
from .utils import Timer


@metrics.fill_dct("IS")
class InceptionScore(BaseGanMetric):
    """
    Class for Inception Score calculation
    """

    def __init__(
            self,
            device = torch.device('cuda:0'),
            splits_n: int = 10,
            **kwargs
        ):
        """
        Args:
            splits_n: number of batchs to calculate IS for each one
            device: what model device to use
        """

        self.device = device
        self.splits_n = splits_n
        self.inception_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[1008]] , resize_input=False)


    def calc_is(
            self,
            probs: torch.Tensor
        ) -> Tuple[float, dict]:
        """
        Batch Inception Score calculation

        Args:
            probs: inception features (conditional probabilities of classes)
        returns:
            inseption score value
            dictionary with time information
        """

        time_info = {}
        with Timer(time_info, "InceptionScore: calc_is"):

            kl = probs * (
                        torch.log(probs) -
                        torch.log(torch.unsqueeze(torch.mean(probs, 0), 0)))
            kl = torch.mean(torch.sum(kl, 1))
            res = torch.exp(kl)

        return res, time_info


    def calc_metric(
            self,
            generated_data: Union[DataLoader, str]
        ) -> Tuple[float, float, dict]:
        """
        Inception Score calculation

        Args:
            train: generated data
            probs: inception features (conditional probabilities of classes)
        returns:
            mean and standard deviation of batchs' IS
            dictionary with time information
        """
        if isinstance(generated_data, DataLoader):
            loader = generated_data
            probs_pth = None
        elif isinstance(str):
            loader = None
            probs_pth = generated_data
        else:
            raise TypeError("InceptionScore.calc_metric: generated_data should be DataLoader or str")

        time_info = {}
        with Timer(time_info, "InceptionScore: calc_metric"):

            if probs_pth is None:
                with Timer(time_info, "InceptionV3: get_features"):
                    probs = self.inception_model.get_features(loader, dim=1008, device=self.device)
            else:
                probs = np.load(probs_pth)

            scores = []
            with Timer(time_info, "InceptionScore: total_calc_is"):
                for i, batch in enumerate(torch.split(probs, split_size_or_sections=probs.shape[0]//self.splits_n), start=0):
                    score = self.calc_is(batch)[0]
                    scores.append(score)

                scores = torch.stack(scores)
                inception_score = torch.mean(scores).cpu().item()
                std = torch.std(scores).cpu().item()

            time_info["duration/InceptionScore: mean_calc_is"] = time_info["duration/InceptionScore: total_calc_is"]/self.splits_n

        return inception_score, std, time_info
