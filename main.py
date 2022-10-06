import argparse
import numpy as np

from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from gan_metrics_in_pytorch import InceptionScore, FID
from gan_metrics_in_pytorch.utils import MyCifarDataset, clean_fid_transformer, ImageDataset

def run(args):
    print(args)

    transformer = clean_fid_transformer if args.clean_fid else T.ToTensor()

    if args.train_data_pth is not None:
        train_dataset = ImageDataset(root=args.train_data_pth, exts=['png'], transform=transformer)
        # train_dataset = MyCifarDataset("./", download=False, train=True, transform=transformer)
        train_data = DataLoader(train_dataset, batch_size=50, num_workers=2)
    if args.train_stats_pth is not None:
        train_data = args.train_stats_pth

    if args.test_data_pth:
        test_dataset = ImageDataset(root=args.test_data_pth, exts=['png'], transform=transformer)
        # test_dataset = MyCifarDataset("./", download=False, train=False, transform=transformer)
        test_data = DataLoader(test_dataset, batch_size=50, num_workers=2)
    if args.test_stats_pth is not None:
        test_data = args.test_stats_pth

    if args.metric=="IS":

        is_class = InceptionScore()
        if not args.test:
            iscore, is_std, time_info = is_class.calc_metric(train_data)
        else:
            iscore, is_std, time_info = is_class.calc_metric(test_data)

        time = time_info["duration/InceptionScore: calc_metric"]
        print(f"IS: {iscore}, IS_std: {is_std}, required_time: {time}")

    elif args.metric=="FID":

        fid_class = FID(
            test_data=test_data,
            use_torch=args.use_torch
        )
        fid, time_info = fid_class.calc_metric(
            train_data=train_data
        )

        time = time_info["duration/FID: calc_metric"]
        print(f"FID: {fid}, required_time: {time}")

    else:
        print("Unknown metric")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate Inception Score or FID")
    parser.add_argument('--metric', type=str, required=True,
                        help='which metric to calculate (IS or FID)')

    parser.add_argument('--train_data_pth', type=str, required=False,
                        help='path to train(generated) data')
    parser.add_argument('--test_data_pth', type=str, required=False,
                        help='path to test(real world) data')

    parser.add_argument('--train_stats_pth', type=str, required=False,
                        help='precalculated generated data statistics')
    parser.add_argument('--test_stats_pth', type=str, required=False,
                        help='precalculated reference statistics')

    parser.add_argument('--test', action='store_true', required=False,
                    help='whether to compute IS on test data')

    parser.add_argument('--clean_fid', action='store_true', required=False,
                    help='using clean fid preprocessing')

    parser.add_argument('--use_torch', action='store_true', required=False,
                    help='using pytorch as the matrix operations backend')
    args = parser.parse_args()

    run(args)
