from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base_class import BaseMetric, metrics
from .datasets import datasets
from .inception_net import InceptionV3


@metrics.add_to_registry("precision_recall")
class ImprovedPRD(BaseMetric):
    """
    Class for Improved Precision and Recall for Distributions (PRD) calculation
    https://arxiv.org/pdf/1904.06991.pdf
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
        k: int = 3,
        **kwargs
    ):
        """
        Args:
            device: what torch device to use for calculations
            k: k for kth-nearest-neighbor
        """
        super().__init__(device)

        self.inception_model = InceptionV3(
            [InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=False
        )
        self.k = k

    def get_knn_distances(self, features: torch.FloatTensor) -> np.array:
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
        a_knn_distances: np.array,
        a_ftrs: torch.Tensor,
        b_ftrs: torch.Tensor,
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

    def compute_ftrs(self, path: str, data_type: str = "folder", save_path=None) -> torch.Tensor:
        """
        Compute Inception Model features of given data

        Args:
            path: path to images or statistics
            data_type: folder, stats or some standard dataset (e. g. cifar10)
            save_path: where to save statistics
        returns:
            ftrs: Inception Model features
        """
        if data_type == "stats":
            ftrs = torch.tensor(np.load(path)).to(self.device)
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

    def __call__(self, gen_path, real_path: str, gen_type="folder", real_type="folder",
                 gen_save_path=None, real_save_path=None) -> Tuple[float, float]:
        """
        Calculates Improved PRD for real and generated data

        Args:
            gen_path: path to generated samples or features from Inception Model
            real_path: path to real samples or features from Inception Model
            gen_type: type of generated data (folder, stats, or standard dataset)
            real_type: type of real data (folder, stats, or standard dataset)
            gen_save_path: path where to save generated Inception Model features
            real_save_path: path where to save real Inception Model features
        returns:
            precision, recall of generated_data w. r. t. real_data
        """
        gen_ftrs = self.compute_ftrs(gen_path, gen_type, gen_save_path)
        real_ftrs = self.compute_ftrs(real_path, real_type, real_save_path)

        real_knn_distances = self.get_knn_distances(real_ftrs)
        gen_knn_distances = self.get_knn_distances(gen_ftrs)

        precision = self.calc_coverage(real_knn_distances, real_ftrs, gen_ftrs)
        recall = self.calc_coverage(gen_knn_distances, gen_ftrs, real_ftrs)

        return precision, recall
