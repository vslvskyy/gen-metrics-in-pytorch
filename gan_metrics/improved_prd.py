from typing import Tuple, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base_class import BaseGanMetricStats, metrics
from .inception_net import InceptionV3
from .utils import datasets, base_transformer


@metrics.add_to_registry("precision_recall")
class ImprovedPRD(BaseGanMetricStats):
    """
    Class for Improved Precision and Recall for Distributions (PRD) calculation
    """

    def __init__(
        self,
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
        super().__init__()

        self.inception_model = InceptionV3(
            [InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=False
        )
        self.transform = base_transformer
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
            pairwise_distances = torch.norm(features - f, 2, dim=-1).cpu().numpy()
            kth_distances[i] = np.partition(pairwise_distances, self.k)[self.k]

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

    def compute_ftrs(self, path, data_type, save_path=None) -> torch.Tensor:
        if data_type == "stats":
            ftrs = np.load(path)
        else:
            data = datasets[data_type](
                root=path,
                transform=self.transform,
                train=True,
                download=True
            )
            data = DataLoader(data, batch_size=50, num_workers=2)
            ftrs = self.inception_model.get_features(data, dim=2048, device=self.device)

            if save_path is not None:
                np.save(save_path, ftrs.cpu().numpy())

        return ftrs

    def __call__(self, gen_path, gen_type="folder", real_path=None, real_type="folder", gen_save_path=None, real_save_path=None) -> Tuple[float, float]:
        """
        Calculates Improved PRD for real and generated data

        Args:
            gen_path: path to generated samples or features from Inception Model
            gen_type: type of generated data (folder, stats, or standard dataset)
            real_path: path to real samples or features from Inception Model
            real_type: type of real data (folder, stats, or standard dataset)
        returns:
            precision, recall of generated_data w. r. t. real_data
        """
        gen_ftrs = self.compute_ftrs(gen_path, gen_type, gen_save_path)
        real_ftrs = self.compute_ftrs(real_path, real_type, real_save_path)
        num_obj = min(len(real_ftrs), len(gen_ftrs))

        real_knn_distances = self.get_knn_distances(real_ftrs[:num_obj])
        gen_knn_distances = self.get_knn_distances(gen_ftrs[:num_obj])

        precision = self.calc_coverage(real_knn_distances, real_ftrs[:num_obj], gen_ftrs[:num_obj])
        recall = self.calc_coverage(gen_knn_distances, gen_ftrs[:num_obj], real_ftrs[:num_obj])

        return precision, recall
