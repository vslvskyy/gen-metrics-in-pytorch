from typing import Tuple, Optional

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

    def get_knn_distances(self, features: torch.Tensor) -> torch.Tensor:
        """
        Calculates distance to kth-neares-neighbor for each feature_vector from given manifold

        Args:
            features: given manifold
        returns:
            distance to kth-neares-neighbor for each feature_vector
        """
        knn_distances = np.zeros(len(features))
        for i, f in enumerate(features):
            pairwise_distances = torch.norm(features - f, 2, dim=-1).cpu().numpy()
            knn_distances[i] = np.partition(pairwise_distances, self.k)[self.k]

        return torch.tensor(knn_distances).to(self.device)

    @staticmethod
    def compute_coverage(
        a_knn_distances: torch.Tensor,
        a_ftrs: torch.Tensor,
        b_ftrs: torch.Tensor,
    ) -> float:
        """
        Calculate ration of samples from manifold B that can be sampled from manifold A

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
            if (ds <= a_knn_distances).any():
                cnt += 1

        return cnt / len(b_ftrs)

    def read_ftrs_from_file(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read features and knn_distances from file

        Args:
            path: path features and knn_distances
        returns:
            ftrs: features
            knn_distances: knn_distances
        """
        f = np.load(path)
        ftrs, knn_distances = f["ftrs"][:], f["knn_distances"][:]
        f.close()
        ftrs = torch.tensor(ftrs, dtype=torch.float).to(self.device)
        knn_distances = torch.tensor(knn_distances, dtype=torch.float).to(self.device)

        return ftrs, knn_distances

    def compute_ftrs_from_data(
        self, data: DataLoader, save_path: Optional[str] = None
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Compute features and knn_distances from DataLoader

        Args:
            data: Dataloader with images
            save_path: where to save statistics
        returns:
            ftrs: features
            knn_distances: knn_distances
        """
        ftrs = self.inception_model.get_features(data, dim=2048, device=self.device)
        knn_distances = self.get_knn_distances(ftrs)

        if save_path is not None:
            np.savez_compressed(save_path, ftrs=ftrs.cpu().numpy(), knn_distances=knn_distances.cpu().numpy())

        return ftrs, knn_distances

    def compute_ftrs(
        self, path: str, data_type: str = "folder", save_path: Optional[str] = None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Inception Model features of given data and pairwise distances for knn

        Args:
            path: path to images or statistics
            data_type: folder, stats or some standard dataset (e. g. cifar10)
            save_path: where to save statistics
        returns:
            ftrs: Inception Model features
            knn_distances: pairwise distances
        """
        if data_type == "stats":
            ftrs, knn_distances = self.read_ftrs_from_file(path)
        else:
            data = datasets[data_type](
                root=path,
                transform=self.transform,
                train=True,
                download=True,
                **kwargs
            )
            data = DataLoader(data, batch_size=50, num_workers=2)
            ftrs, knn_distances = self.compute_ftrs_from_data(data, save_path)

        return ftrs, knn_distances

    def __call__(
        self, gen_path: str, real_path: str, gen_type: str = "folder", real_type: str = "folder",
        gen_save_path: Optional[str] = None, real_save_path: Optional[str] = None, **kwargs
    ) -> Tuple[float, float]:
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
        gen_ftrs, gen_knn_distances = self.compute_ftrs(gen_path, gen_type, gen_save_path, **kwargs)
        real_ftrs, real_knn_distances = self.compute_ftrs(real_path, real_type, real_save_path, **kwargs)

        precision = self.compute_coverage(real_knn_distances, real_ftrs, gen_ftrs)
        recall = self.compute_coverage(gen_knn_distances, gen_ftrs, real_ftrs)

        return precision, recall
