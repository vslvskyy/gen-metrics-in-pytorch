import argparse
import numpy as np

from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from gan_metrics_in_pytorch import InceptionScore, FID
from gan_metrics_in_pytorch.utils import MyCifarDataset, clean_fid_transformer

def run(args):

    transformer = clean_fid_transformer

    train_dataset = MyCifarDataset("./", download=True, train=True, transform=transformer)
    train_loader = DataLoader(train_dataset, batch_size=50, num_workers=2)

    test_dataset = MyCifarDataset("./", download=True, train=False, transform=transformer)
    test_loader = DataLoader(test_dataset, batch_size=50, num_workers=2)

    if args.metric=='IS':
        is_class = InceptionScore(train_loader)
        iscore, is_std, time_info = is_class.calc_metric()
        time = time_info["duration/InceptionScore: calc_metric"]
        print(f"IS: {iscore}, IS_std: {is_std}, required_time: {time}")
    elif args.metric=='FID':
        fid_class = FID(train_loader, test_loader)
        fid, time_info = fid_class.calc_metric(args.train_stats_pth, args.test_stats_pth, use_torch=args.use_torch)
        time = time_info["duration/FID: calc_metric"]
        print(f"FID: {fid}, required_time: {time}")
    else:
        print("Unknown metric")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate Inception Score or FID")
    parser.add_argument('--metric', type=str, required=True,
                        help='which metric to calculate (IS or FID)')
    # parser.add_argument('--train_data_pth', type=str, required=False,
    #                     help='path to train(generated) data')
    # parser.add_argument('--test_data_pth', type=str, required=False,
    #                     help='path to test(real world) data')
    parser.add_argument('--train_stats_pth', type=str, required=False,
                        help='precalculated generated data statistics')
    parser.add_argument('--test_stats_pth', type=str, required=False,
                        help='precalculated reference statistics')
    parser.add_argument('--use_torch', action='store_true', required=False,
                    help='using pytorch as the matrix operations backend')
    args = parser.parse_args()

    run(args)
