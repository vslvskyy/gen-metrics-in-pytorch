from gan_metrics import get_inception_score, get_fid
from gan_metrocs.frechet_inception_distance import calculate_frechet_distance

import numpy as np
import pandas as pd
from torch import nn
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader

from typing import Dict, Any
import time
from time import time
import collections
from collections import defaultdict


def calc_metrics(loader, metric_dct: defaultdict, time_dct: defaultdict, data_type: str = "train", use_torch: bool = False):

    start_time = time()
    IS, IS_std = get_inception_score(loader, use_torch=use_torch)
    time_dct[f"use_torch={use_torch}"].append(time() - start_time)
    metric_dct[f"use_torch={use_torch}"].extend([IS, IS_std])

    if data_type == "train":
        start_time = time()
        FID = get_fid(loader, 'cifar10.test.npz', use_torch=use_torch)
        time_dct[f"use_torch={use_torch}"].append(time() - start_time)
        metric_dct[f"use_torch={use_torch}"].append(FID)

class MyDataset(CIFAR10):
    def __getitem__(self, idx):
        return super().__getitem__(idx)[0]


def main():
    metric_dct = defaultdict(list)
    time_dct = defaultdict(list)
    for use_torch in [True, False]:
        calc_metrics(train_loader, metric_dct, time_dct, use_torch=use_torch)
        calc_metrics(test_loader, metric_dct, time_dct, data_type="test", use_torch=use_torch)


    f = np.load("cifar10.train.npz")
    mu_train, sigma_train = f['mu'][:], f['sigma'][:]
    f.close()
    f = np.load("cifar10.test.npz")
    mu_test, sigma_test = f['mu'][:], f['sigma'][:]
    f.close()

    for use_torch in [True, False]:
    start_time = time()
    FID = calculate_frechet_distance(
              torch.tensor(mu_train), torch.tensor(sigma_train),
              torch.tensor(mu_test), torch.tensor(sigma_test),
              use_torch=use_torch
          )
    time_dct[f"use_torch={use_torch}"].append(time() - start_time)
    metric_dct[f"use_torch={use_torch}"].append(FID)

    np.save("our_metric_dct.npy", metric_dct)
    np.save("our_time_dct.npy", time_dct)



if __name__ == '__main__':
    main()
