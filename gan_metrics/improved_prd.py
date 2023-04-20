from typing import Union, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base_class import BaseGanMetricStats, metrics
from .inception_net import InceptionV3
from .utils import Timer


@metrics.fill_dct("ImprovedPRD")
class ImprovedPRD(BaseGanMetricStats):
    """
    Class for Improved Precision and Recall for Distributions (PRD) calculation
    """

    def __init__(
        self,
        real_data: DataLoader,
        device: torch.device = torch.device("cuda:0"),
        k: int = 3,
        # добавить возможность считать признаки из vgg-16
        **kwargs
    ):
        """
        Args:
            real_data: real world data
            device: what model device to use
            k: k for kth-nearest-neighbor
        """
        super().__init__(real_data)
        if self.real_data_loader is None:
            raise TypeError("ImprovedPRD.__init__: real_data should be DataLoader")

        self.inception_model = InceptionV3(
            [InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=False
        )
        self.real_ftrs = None
        self.real_knn_distances = None
        self.device = device
        self.k = k

    def get_knn_distances(self, features: torch.FloatTensor) -> Dict[int, float]:
        """
        Calculates distance to kth-neares-neighbor for each feature_vector from given manifold

        Args:
            features: given manifold
        returns:
            distance to kth-neares-neighbor for each feature_vector
        """
        kth_distances = np.zeros(len(features))
        for i, f in enumerate(features):
            pairwise_distances = torch.norm(features - f, 2, dim=-1)
            kth_distances[i] = np.partition(pairwise_distances.cpu().numpy(), self.k)[
                self.k
            ]

        return kth_distances

    def calc_coverage(
        self,
        a_knn_distances: Dict[int, float],
        a_ftrs: torch.FloatTensor,
        b_ftrs: torch.FloatTensor,
    ) -> float:
        """
        Calculates ration of samples from manifold B that can be sampled from manifold A

        Args:
            a_knn_distances: distances to kth-nearest-neighbor for all samples from A
            a_ftrs: manifold A
            b_ftrs: manifold B
        returns:
            ration of samples from manifold B that can be sampled from manifold A
        """
        cnt = 0
        for b in b_ftrs:
            ds = torch.norm(a_ftrs - b, 2, dim=-1)
            if (ds.cpu().numpy() <= a_knn_distances).any():
                cnt += 1

        return cnt / len(b_ftrs)

    def calc_metric(
        self, generated_data: DataLoader, precision: bool = True, recall: bool = True
    ) -> Tuple[Union[Tuple[float, float], float], dict]:
        """
        Calculates Improved PRD for real and generated data

        Args:
            generated_data: dataloader with generated samples
            precision: wether to calculate precision
            recall: wether to calculate recall
        returns:
            precision, recall of generated_data w. r. t. real_data
        """

        if isinstance(generated_data, DataLoader):
            generated_data_loader = generated_data
        else:
            raise TypeError(
                "ImprovedPRD.calc_metric: generated_data should be DataLoader"
            )

        time_info = {}
        with Timer(time_info, "ImprovedPRD: calc_metric"):
            if self.real_knn_distances is None:
                if self.real_ftrs is None:
                    with Timer(time_info, "InceptionV3: get_features (real)"):
                        self.real_ftrs = self.inception_model.get_features(
                            self.real_data_loader, dim=2048, device=self.device
                        )

                with Timer(time_info, "ImprovedPRD: get_knn_distances (real)"):
                    self.real_knn_distances = self.get_knn_distances(self.real_ftrs)

            with Timer(time_info, "InceptionV3: get_features (generated)"):
                generated_ftrs = self.inception_model.get_features(
                    generated_data_loader, dim=2048, device=self.device
                )

            num_obj = min(
                len(self.real_ftrs), len(generated_ftrs)
            )  # to be sure, that |real_ftrs| = |generated_ftrs|
            if num_obj <= 0:
                raise TypeError(
                    "ImprovedPRD.calc_metric: generated or real data manifold is empty"
                )

            with Timer(time_info, "ImprovedPRD: get_knn_distances (generated)"):
                self.generated_knn_distances = self.get_knn_distances(
                    generated_ftrs[:num_obj]
                )

            if precision:
                with Timer(time_info, "ImprovedPRD: calc_coverage (precision)"):
                    precision = self.calc_coverage(
                        self.real_knn_distances[:num_obj],
                        self.real_ftrs[:num_obj],
                        generated_ftrs[:num_obj],
                    )

            if recall:
                with Timer(time_info, "ImprovedPRD: calc_coverage (recall)"):
                    recall = self.calc_coverage(
                        self.generated_knn_distances,
                        generated_ftrs[:num_obj],
                        self.real_ftrs[:num_obj],
                    )

        if precision and recall:
            return (precision, recall), time_info
        if precision:
            return precision, time_info
        if recall:
            return recall, time_info
