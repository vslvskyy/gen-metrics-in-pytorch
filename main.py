import argparse
import numpy as np

from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from gan_metrics_in_pytorch import InceptionScore, FID

def run(args):
    # dataset = ImageFolder(args.path, transform=T.ToTensor())
    # loader = DataLoader(dataset, batch_size=50, num_workers=4)
    loader = np.load(args.path_to_loader, allow_pickle=True) # не работает

    if args.metric=='IS':
        is_class = InceptionScore(loader)
        iscore, is_std, time_info = is_class.calc_metric()
        time = time_info["duration/InceptionScore: calc_metric"]
        print(f"IS: {iscore}, IS_std: {is_std}, required_time: {time}")
    elif args.metric=='FID':
        FID = get_fid(loader, args.stats, use_torch=args.use_torch)
        print(FID)
    else:
        print("Unknown metric")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate Inception Score or FID")
    parser.add_argument('--path_to_loader', type=str, required=True,
                        help='path to train dataloader')
    parser.add_argument('--metric', type=str, required=True,
                        help='which metric to calculate (IS or FID)')
    parser.add_argument('--stats', type=str, required=False,
                        help='precalculated reference statistics')
    parser.add_argument('--use_torch', action='store_true', required=False,
                    help='using pytorch as the matrix operations backend')
    args = parser.parse_args()

    run(args)
