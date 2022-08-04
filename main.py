import argparse

from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from .gan_metrics import get_inception_score, get_fid


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate Inception Score or FID")
    parser.add_argument('--use_torch', action='store_true',
                        help='using pytorch as the matrix operations backend')
    parser.add_argument('--path', type=str, required=True,
                        help='path to image directory')
    parser.add_argument('--metric', type=str, required=True,
                        help='which metric to calculate (IS or FID)')
    args = parser.parse_args()
    if args.metric=='FID':
        parser.add_argument('--stats', type=str, required=True,
                            help='precalculated reference statistics')
    args = parser.parse_args()

    dataset = ImageFolder(args.path, transform=T.ToTensor())
    loader = DataLoader(dataset, batch_size=50, num_workers=4)

    if args.metric=='IS':
        IS, IS_std = get_inception_score(loader, use_torch=args.use_torch)
        print(IS, IS_std)
    elif args.metric=='FID':
        FID = get_fid(loader, args.stats, use_torch=args.use_torch)
        print(FID)

    # (IS, IS_std), FID = get_inception_score_and_fid(
    #     loader, args.stats, use_torch=args.use_torch, verbose=True)
    # print(IS, IS_std, FID)
