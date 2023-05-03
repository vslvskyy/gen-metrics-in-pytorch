from typing import Optional, Tuple

import numpy as np
import torch
from scipy import linalg
from torch.utils.data import DataLoader

from .base_class import BaseMetric, metrics
from .datasets import datasets
from .inception_net import InceptionV3
from .utils import torch_cov, sqrt_newton_schulz, clean_fid_transformer


@metrics.add_to_registry("fid")
class FID(BaseMetric):
    """
    Class for Frechet Inception Distance calculation
    https://arxiv.org/pdf/1706.08500.pdf
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
        **kwargs
    ):
        """
        Args:
            device: what torch device to use for calculations
        """
        super().__init__(device)
        self.inception_model = InceptionV3(
            [InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=False
        )

    @staticmethod
    def compute_fid_from_stats(
        mu: torch.Tensor, sigma: torch.Tensor,
        ref_mu: torch.Tensor, ref_sigma: torch.Tensor,
        eps: float = 1e-6
    ) -> float:
        """
        Calculates Frechet Distance betweet two distributions

        Args:
            mu: mean of first distribution
            sigma: covariance matrix of first distribution
            ref_mu: mean of second distribution
            ref_sigma: covariance matrix of second distribution
        returns:
            frechet distance between two distributions
        """
        diff = mu - ref_mu
        # Run 50 iterations of newton-schulz to get the matrix sqrt of (sigma1 x sigma2)
        covmean = sqrt_newton_schulz(sigma.mm(ref_sigma).unsqueeze(0), 50).squeeze()
        if not torch.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            print(msg)
            offset = torch.eye(sigma.shape[0]) * eps
            covmean = sqrt_newton_schulz((sigma + offset).mm(ref_sigma + offset).unsqueeze(0), 50).squeeze()

        if not torch.isfinite(covmean).all():
            return float("nan")

        fid = (
            diff.dot(diff) +
            torch.trace(sigma) +
            torch.trace(ref_sigma) -
            2 * torch.trace(covmean)
        ).cpu().item()
        return fid

    def read_stats_from_file(self, path: str) -> Tuple[torch.tensor, torch.tensor]:
        """
        Read mean vector mu and covariance matrix sigma from file

        Args:
            path: path to file with statistics
        returns:
            mu: mean vector
            sigma: covariance matrix
        """
        f = np.load(path)
        mu, sigma = f["mu"][:], f["sigma"][:]
        f.close()
        mu = torch.tensor(mu, dtype=torch.float).to(self.device)
        sigma = torch.tensor(sigma, dtype=torch.float).to(self.device)

        return mu, sigma

    def compute_stats_from_data(
        self, data: DataLoader, save_path: Optional[str] = None
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Compute mean vector mu and covariance matrix sigma from DataLoader

        Args:
            data: Dataloader with images
            save_path: where to save statistics
        returns:
            mu: mean vector
            sigma: covariance matrix
        """
        acts = self.inception_model.get_features(data, dim=2048, device=self.device)
        mu = torch.mean(acts, axis=0)
        sigma = torch_cov(acts)

        if save_path is not None:
            np.savez_compressed(save_path, mu=mu.cpu().numpy(), sigma=sigma.cpu().numpy())

        return mu, sigma

    def compute_stats(self, path: str, data_type: str = "folder", save_path=None) -> torch.Tensor:
        """
        Compute mean vector mu and covariance matrix sigma of given data

        Args:
            path: path to images or statistics
            data_type: folder, stats or some standard dataset (e. g. cifar10)
            save_path: where to save statistics
        returns:
            mu: mean vector
            sigma: covariance matrix
        """
        if data_type == "stats":
            mu, sigma = self.read_stats_from_file(path)
        else:
            data = datasets[data_type](
                root=path,
                transform=self.transform,
                train=True,
                download=True
            )
            data = DataLoader(data, batch_size=50, num_workers=2)
            mu, sigma = self.compute_stats_from_data(data, save_path)

        return mu, sigma

    def __call__(
        self, gen_path: str, real_path: str, gen_type: str = "folder", real_type: str = "folder",
        gen_save_path: Optional[str] = None, real_save_path: Optional[str] = None, eps: float = 1e-6
    ) -> float:
        """
        Calculates Frecher Inception Distance for real and generated data

        Args:
            gen_path: path to generated samples or features from Inception Model
            gen_type: type of generated data (folder, stats, or standard dataset)
            real_path: path to real samples or features from Inception Model
            real_type: type of real data (folder, stats, or standard dataset)
            gen_save_path: path where to save generated Inception Model statistics
            real_save_path: path where to save real Inception Model statistics
        returns:
            frechet inception distance between real and generated data
        """
        mu, sigma = self.compute_stats(gen_path, gen_type, gen_save_path)
        ref_mu, ref_sigma = self.compute_stats(real_path, real_type, real_save_path)
        return self.compute_fid_from_stats(mu, sigma, ref_mu, ref_sigma, eps)


@metrics.add_to_registry("clean_fid")
class CLEANFID(FID):
    """
    Class for Clean Frechet Inception Distance calculation
    https://arxiv.org/pdf/2104.11222.pdf
    """

    def __init__(
            self,
            device: torch.device = torch.device("cuda:0"),
            **kwargs
        ):
        super().__init__(device)
        self.transform = clean_fid_transformer

    def read_stats_from_file(self, path):
        print("Please, be sure that these stats are calculated from data with correct transform!")
        return super().read_stats_from_file(path)


@metrics.add_to_registry("fid_numpy")
class FIDNUMPY(FID):
    """
    Class for Frechet Inception Distance calculation in numpy
    https://arxiv.org/pdf/1706.08500.pdf
    """

    @staticmethod
    def compute_fid_from_stats(
        mu: torch.Tensor, sigma: torch.Tensor,
        ref_mu: torch.Tensor, ref_sigma: torch.Tensor,
        eps: float = 1e-6
    ) -> float:
        """
        Calculates Frechet Distance betweet two distributions in numpy

        Args:
            mu: mean of first distribution
            sigma: covariance matrix of first distribution
            ref_mu: mean of second distribution
            ref_sigma: covariance matrix of second distribution

        returns:
            frechet distance between two distributions
        """
        mu = np.atleast_1d(mu.cpu().numpy())
        ref_mu = np.atleast_1d(ref_mu.cpu().numpy())

        sigma = np.atleast_2d(sigma.cpu().numpy())
        ref_sigma = np.atleast_2d(ref_sigma.cpu().numpy())

        diff = mu - ref_mu

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma.dot(ref_sigma), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            print(msg)
            offset = np.eye(sigma.shape[0]) * eps
            covmean = linalg.sqrtm((sigma + offset).dot(ref_sigma + offset))
        if not np.isfinite(covmean).all():
            return float("nan")
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        fid = (
            diff.dot(diff) +
            np.trace(sigma) +
            np.trace(ref_sigma) -
            2 * np.trace(covmean)
        )

        return fid
