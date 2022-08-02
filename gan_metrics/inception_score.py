from typing import List, Union, Tuple

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from .inception_net import InceptionV3


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_inception_feature(
    images: Union[List[torch.FloatTensor], DataLoader],
    dims: List[int],
    batch_size: int = 50,
    use_torch: bool = False,
    verbose: bool = False,
    device: torch.device = torch.device('cuda:0'),
) -> Union[torch.FloatTensor, np.ndarray]:
    """Calculate Inception Score and FID.
    For each image, only a forward propagation is required to
    calculating features for FID and Inception Score.
    Args:
        images: List of tensor or torch.utils.data.Dataloader. The return image
            must be float tensor of range [0, 1].
        dims: List of int, see InceptionV3.BLOCK_INDEX_BY_DIM for
            available dimension.
        batch_size: int, The batch size for calculating activations. If
            `images` is torch.utils.data.Dataloader, this argument is
            ignored.
        use_torch: bool. The default value is False and the backend is same as
            official implementation, i.e., numpy. If use_torch is enableb,
            the backend linalg is implemented by torch, the results are not
            guaranteed to be consistent with numpy, but the speed can be
            accelerated by GPU.
        verbose: Set verbose to False for disabling progress bar. Otherwise,
            the progress bar is showing when calculating activations.
        device: the torch device which is used to calculate inception feature
    Returns:
        inception_score: float tuple, (mean, std)
        fid: float
    """
    assert all(dim in InceptionV3.BLOCK_INDEX_BY_DIM for dim in dims)

    is_dataloader = isinstance(images, DataLoader)
    if is_dataloader:
        num_images = min(len(images.dataset), images.batch_size * len(images))
        batch_size = images.batch_size
    else:
        num_images = len(images)

    block_idxs = [InceptionV3.BLOCK_INDEX_BY_DIM[dim] for dim in dims]
    model = InceptionV3(block_idxs).to(device)
    model.eval()

    if use_torch:
        features = [torch.empty((num_images, dim)).to(device) for dim in dims]
    else:
        features = [np.empty((num_images, dim)) for dim in dims]

    pbar = tqdm(
        total=num_images, dynamic_ncols=True, leave=False,
        disable=not verbose, desc="get_inception_feature")
    looper = iter(images)
    start = 0
    while start < num_images:
        # get a batch of images from iterator
        if is_dataloader:
            batch_images = next(looper)
        else:
            batch_images = images[start: start + batch_size]
        end = start + len(batch_images)

        # calculate inception feature
        batch_images = batch_images.to(device)
        with torch.no_grad():
            outputs = model(batch_images)
            for feature, output, dim in zip(features, outputs, dims):
                if use_torch:
                    feature[start: end] = output.view(-1, dim)
                else:
                    feature[start: end] = output.view(-1, dim).cpu().numpy()
        start = end
        pbar.update(len(batch_images))
    pbar.close()
    return features
    

def calculate_inception_score(
    probs: Union[torch.FloatTensor, np.ndarray],
    splits: int = 10,
    use_torch: bool = False,
) -> Tuple[float, float]:
    # Inception Score
    scores = []
    for i in range(splits):
        part = probs[
            (i * probs.shape[0] // splits):
            ((i + 1) * probs.shape[0] // splits), :]
        if use_torch:
            kl = part * (
                torch.log(part) -
                torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
            kl = torch.mean(torch.sum(kl, 1))
            scores.append(torch.exp(kl))
        else:
            kl = part * (
                np.log(part) -
                np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
    if use_torch:
        scores = torch.stack(scores)
        inception_score = torch.mean(scores).cpu().item()
        std = torch.std(scores).cpu().item()
    else:
        inception_score, std = (np.mean(scores), np.std(scores))
    del probs, scores
    return inception_score, std


def get_inception_score(
    images: Union[torch.FloatTensor, DataLoader],
    splits: int = 10,
    use_torch: bool = False,
    **kwargs,
) -> Tuple[Tuple[float, float], float]:
    """Calculate Inception Score.
    Please refer to `get_inception_score_and_fid` for the arguments
    descriptions.
    Returns:
        Inception Score: float tuple
    """
    probs, = get_inception_feature(
        images, dims=[1008], use_torch=use_torch, **kwargs)
    inception_score, std = calculate_inception_score(probs, splits, use_torch)
    return (inception_score, std)

