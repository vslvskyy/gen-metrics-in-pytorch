from typing import Union, Tuple

import numpy as np
import torch
from scipy import linalg
from torch.utils.data import DataLoader

from .base_class import BaseGanMetricStats, metrics
from .inception_net import InceptionV3
from .utils import Timer, torch_cov, sqrt_newton_schulz


@metrics.fill_dct("FID")
class FID(BaseGanMetricStats):
    """
    Class for Frechet Inception Distance calculation
    """

    def __init__(
            self,
            real_data: Union[DataLoader, str],
            use_torch: bool = False,
            device: torch.device = torch.device('cuda:0'),
            eps: float = 1e-6,
            **kwargs
        ):
        """
        Args:
            real_data: real world data
            is_clean_fid: whether to use clean-fid preprocessing
            use_torch: whether to use pytorch or numpy for frechet distance calculation
            device: what model device to use
            eps: prevent covmean from being singular matrix
        """
        super().__init__(real_data)
        self.inception_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=False)
        self.use_torch = use_torch
        self.device = device
        self.eps = eps

    def calc_stats(
            self,
            loader: DataLoader,
            device: torch.device = torch.device('cuda:0')
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, dict]:
        """
        Calculates statistics of inception features

        Args:
            loader: real world or generated data
            batch_size: number of objects in one batch
            device: what model device to use
        returns:
            mean and covariance of inception features
            dictionary with time information

        """

        time_info = {}
        with Timer(time_info, "FID: calc_stats"):
            with Timer(time_info, "InceptionV3: get_features"):
                acts = self.inception_model.get_features(loader, dim=2048, device=device)
            mu = torch.mean(acts, axis=0)
            sigma = torch_cov(acts)
        return mu, sigma, time_info

    def calc_fd_in_np(
            self,
            m1: np.ndarray,
            s1: np.ndarray,
            m2: np.ndarray,
            s2: np.ndarray
        ) -> Tuple[float, dict]:
        """
        Calculates Frechet Distance betweet two distributions

        Args:
            m1: mean of first distribution
            s1: covariance matrix of first distribution
            m2: mean of second distribution
            s2: covariance matrix of second distribution
        returns:
            frechet distance between two distributions
            dictionry with time information
        """
        time_info = {}
        with Timer(time_info, "FID: calc_fd_in_np"):
            m1 = np.atleast_1d(m1)
            m2 = np.atleast_1d(m2)

            s1 = np.atleast_2d(s1)
            s2 = np.atleast_2d(s2)

            diff = m1 - m2

            # Product might be almost singular
            covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
            if not np.isfinite(covmean).all():
                msg = ('fid calculation produces singular product; '
                    'adding %s to diagonal of cov estimates') % self.eps
                print(msg)
                offset = np.eye(s1.shape[0]) * self.eps
                covmean = linalg.sqrtm((s1 + offset).dot(s2 + offset))

            # Numerical error might give slight imaginary component
            if np.iscomplexobj(covmean):
                if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                    m = np.max(np.abs(covmean.imag))
                    raise ValueError("Imaginary component {}".format(m))
                covmean = covmean.real

            tr_covmean = np.trace(covmean)

            fd = (diff.dot(diff) +
                np.trace(s1) +
                np.trace(s2) -
                2 * tr_covmean)
        return fd, time_info

    def calc_fd_in_torch(
            self,
            m1: torch.Tensor,
            s1: torch.Tensor,
            m2: torch.Tensor,
            s2: torch.Tensor
        ) -> Tuple[float, dict]:
        """
        Calculates Frechet Distance betweet two distributions

        Args:
            m1: mean of first distribution
            s1: covariance matrix of first distribution
            m2: mean of second distribution
            s2: covariance matrix of second distribution
        returns:
            frechet distance between two distributions
            dictionry with time information
        """

        time_info = {}
        with Timer(time_info, "FID: calc_fd_in_torch"):
            diff = m1 - m2
            # Run 50 itrs of newton-schulz to get the matrix sqrt of (sigma1 x sigma2)
            covmean = sqrt_newton_schulz(s1.mm(s2).unsqueeze(0), 50)
            if torch.any(torch.isnan(covmean)):
                return float('nan'), time_info
            covmean = covmean.squeeze()
            fd = (diff.dot(diff) +
                torch.trace(s1) +
                torch.trace(s2) -
                2 * torch.trace(covmean)).cpu().item()
        return fd, time_info

    def calc_metric(
            self,
            generated_data: Union[DataLoader, str]
        ) -> Tuple[float, dict]:
        """
        Calculates Frechet Inception Distance betweet two datasets

        Args:
            generated_data: generated data
        returns:
            frechet inception distance
            dictionary with time information
        """
        if isinstance(generated_data, DataLoader):
            generated_data_loader = generated_data
            generated_data_stats_pth = None
        elif isinstance(generated_data, str):
            generated_data_loader = None
            generated_data_stats_pth = generated_data
        else:
            raise TypeError("FID.calc_metric: generated_data should be DataLoader or str")

        time_info = {}
        with Timer(time_info, "FID: calc_metric"):

            if generated_data_stats_pth is None:
                with Timer(time_info, "FID: calc_stats, generated data"):
                    m1, s1, calc_stats_time = self.calc_stats(generated_data_loader, device=self.device)
                time_info.update(calc_stats_time)

            else:
                f = np.load(generated_data_stats_pth)
                m1, s1 = f['mu'][:], f['sigma'][:]
                f.close()
                if isinstance(m1, type(np.zeros(3))):
                    m1 = torch.tensor(m1, dtype=torch.float).to(self.device)
                if isinstance(s1, type(np.zeros(3))):
                    s1 = torch.tensor(s1, dtype=torch.float).to(self.device)

            if self.real_data_stats_pth is None:
                with Timer(time_info, "FID: calc_stats, real data"):
                    m2, s2, calc_stats_time = self.calc_stats(self.real_data_loader, device=self.device)
                time_info.update(calc_stats_time)

            else:
                f = np.load(self.real_data_stats_pth)
                m2, s2 = f['mu'][:], f['sigma'][:]
                f.close()
                if isinstance(m2, type(np.zeros(3))):
                    m2 = torch.tensor(m2, dtype=torch.float).to(self.device)
                if isinstance(s2, type(np.zeros(3))):
                    s2 = torch.tensor(s2, dtype=torch.float).to(self.device)

            assert m1.shape == m2.shape, \
                'FID.calc_metric: generated and real mean vectors have different lengths'
            assert s1.shape == s2.shape, \
                'FID.calc_metric: generated and real covariances have different dimensions'

            if self.use_torch:
                fid = self.calc_fd_in_torch(m1, s1, m2, s2)[0]
            else:
                fid = self.calc_fd_in_np(m1.cpu().numpy(), s1.cpu().numpy(), m2.cpu().numpy(), s2.cpu().numpy())[0]

        return fid, time_info
