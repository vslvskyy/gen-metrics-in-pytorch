import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Union

from .base_class import BaseGanMetric
from .inception_net import InceptionV3


class InceptionScore(BaseGanMetric):

    def get_is(
        self,
        probs: np.ndarray
    ) -> float:

        kl = probs * (
            np.log(probs) -
            np.log(np.expand_dims(np.mean(probs, 0), 0))
            )
        kl = np.mean(np.sum(kl, 1))

        return np.exp(kl)


    def calc_metric(
        self,
        probs: Optional[Union[torch.FloatTensor, np.ndarray]] = None,
        splits: int = 10,
        use_torch: bool = False,
        device = torch.device('cuda:0'),
    ) -> Tuple[float, float]:
        # Inception Score
        if probs is None:
            model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[1008]]).to(device)
            probs = model.get_features(self.train_loader, dim=1008, use_torch=use_torch)

        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()

        scores = []
        for i in range(splits):
            part = probs[
                (i * probs.shape[0] // splits):
                ((i + 1) * probs.shape[0] // splits), :]

            scores.append(self.get_is(part))

        inception_score, std = (np.mean(scores), np.std(scores))

        del probs, scores
        return inception_score, std
