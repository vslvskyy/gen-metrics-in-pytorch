from typing import List, Union, Tuple

import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm
from torch.utils.data import DataLoader

from .base_class import BaseGanMetricStats
from .inception_net import InceptionV3
from .utils import Timer

def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

# Pytorch implementation of matrix sqrt, from Tsung-Yu Lin, and Subhransu Maji
# https://github.com/msubhransu/matrix-sqrt
def sqrt_newton_schulz(A, numIters, dtype=None):
    with torch.no_grad():
        if dtype is None:
            dtype = A.type()
        batchSize = A.shape[0]
        dim = A.shape[1]
        normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
        Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
        K = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1)
        Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1)
        K = K.type(dtype)
        Z = Z.type(dtype)
        for i in range(numIters):
            T = 0.5 * (3.0 * K - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)
        sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA


class FID(BaseGanMetricStats):

    """
    Class for Frechet Inception Distance calculation
    """

    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader
        ):
        """
        Args:
            train_loader: generated data
            test_loader: real world data
        """

        super().__init__(train_loader, test_loader)
        self.inception_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]])

    def calc_stats(
        self,
        loader: torch.utils.data.DataLoader,
        batch_size: int=50,
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
        s2: np.ndarray,
        eps: float = 1e-6
    ) -> Tuple[float, dict]:
        """
        Calculates Frechet DIstance betweet two distributions

        Args:
            m1: mean of first distribution
            s1: covariance matrix of first distribution
            m2: mean of second distribution
            s2: covariance matrix of second distribution
            eps: prevent covmean from being singular matrix
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
                    'adding %s to diagonal of cov estimates') % eps
                print(msg)
                offset = np.eye(s1.shape[0]) * eps
                covmean = linalg.sqrtm((s1 + offset).dot(s2 + offset))

            # Numerical error might give slight imaginary component
            if np.iscomplexobj(covmean):
                if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                    m = np.max(np.abs(covmean.imag))
                    raise ValueError('Imaginary component {}'.format(m))
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
        s2: torch.Tensor,
        eps: float = 1e-6
    ) -> Tuple[float, dict]:
        """
        Calculates Frechet DIstance betweet two distributions

        Args:
            m1: mean of first distribution
            s1: covariance matrix of first distribution
            m2: mean of second distribution
            s2: covariance matrix of second distribution
            eps: prevent covmean from being singular matrix
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
                return float('nan')
            covmean = covmean.squeeze()
            fd = (diff.dot(diff) +
                torch.trace(s1) +
                torch.trace(s2) -
                2 * torch.trace(covmean)).cpu().item()
        return fd, time_info

    def calc_metric(
        self,
        train_stats_pth: str = None,
        test_stats_pth: str = None,
        use_torch: bool = False,
        eps: float = 1e-6
    ) -> Tuple[float, dict]:
        """
        Calculates Frechet Inception Distance betweet two datasets

        Args:
            train_stats_pth: path to generated data statistics
            test_stats_pth: path to real world data statistics
            use_torch: whethere to use pytorch or numpy for frechet distance calculation
            eps: prevent covmean from being singular matrix
        returns:
            frechet inception distance
            dictionary with time information
        """

        time_info = {}
        with Timer(time_info, "FID: calc_metric"):

            if train_stats_pth is None:
                with Timer(time_info, "FID: calc_stats, train"):
                    m1, s1, calc_stats_time = self.calc_stats(self.train_loader)
                time_info.update(calc_stats_time)
            else:
                f = np.load(train_stats_pth)
                m1, s1 = f['mu'][:], f['sigma'][:]
                f.close()

            if test_stats_pth is None:
                with Timer(time_info, "FID: calc_stats, test"):
                    m2, s2, calc_stats_time = self.calc_stats(self.test_loader)
                time_info.update(calc_stats_time)
            else:
                f = np.load(test_stats_pth)
                m2, s2 = f['mu'][:], f['sigma'][:]
                f.close()

            assert m1.shape == m2.shape, \
                'Training and test mean vectors have different lengths'
            assert s1.shape == s2.shape, \
                'Training and test covariances have different dimensions'

            if use_torch:
                fid = self.calc_fd_in_torch(m1, s1, m2, s2, eps)[0]
            else:
                fid = self.calc_fd_in_np(m1.cpu().numpy(), s1.cpu().numpy(), m2.cpu().numpy(), s2.cpu().numpy(), eps)[0]

        return fid, time_info
