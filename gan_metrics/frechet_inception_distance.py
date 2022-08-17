from typing import List, Union, Tuple

import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm
from torch.utils.data import DataLoader

from .base_class import BaseGanMetricComp
from .inception_net import InceptionV3

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


class FID(BaseGanMetricComp):

    def calc_stats(self, loader, batch_size=50, use_torch=False, device: torch.device = torch.device('cuda:0')): # add loader
        model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)
        acts = model.get_features(loader, dim=2048, use_torch=use_torch)
        if use_torch:
            mu = torch.mean(acts, axis=0)
            sigma = torch_cov(acts)
        else:
            mu = np.mean(acts, axis=0)
            sigma = np.cov(acts, rowvar=False)
        return (mu, sigma)

    def calc_fd_in_np(
        self,
        m1: np.ndarray,
        s1: np.ndarray,
        m2: np.ndarray,
        s2: np.ndarray,
        eps: float = 1e-6
    ) -> float:

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
        return fd

    def calc_fd_in_torch(
        self,
        m1: torch.Tensor,
        s1: torch.Tensor,
        m2: torch.Tensor,
        s2: torch.Tensor,
        eps: float = 1e-6
    ) -> float:

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
        return fd

    def calc_metric(
        self,
        train_stats_pth: str = None,
        test_stats_pth: str = None,
        use_torch: bool = False,
        eps: float = 1e-6
    ) -> float:

        if train_stats_pth is None:
            m1, s1 = self.calc_stats(self.train_loader, use_torch=use_torch)
        else:
            f = np.load(train_stats_pth)
            m1, s1 = f['mu'][:], f['sigma'][:]
            f.close()

        if test_stats_pth is None:
            m2, s2 = self.calc_stats(self.test_loader, use_torch=use_torch)
        else:
            f = np.load(test_stats_pth)
            m2, s2 = f['mu'][:], f['sigma'][:]
            f.close()

        assert m1.shape == m2.shape, \
            'Training and test mean vectors have different lengths'
        assert s1.shape == s2.shape, \
            'Training and test covariances have different dimensions'

        if use_torch:
            return self.calc_fd_in_torch(m1, s1, m2, s2)
        return self.calc_fd_in_np(m1, s1, m2, s2)
