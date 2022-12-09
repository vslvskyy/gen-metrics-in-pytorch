import argparse
from typing import Dict

from torch.utils.data import DataLoader, Dataset

from gan_metrics.base_class import metrics
from gan_metrics.utils import (
    base_transformer,
    clean_fid_transformer,
    datasets,
    Timer
)


def run(args):
    time_info: Dict[str, float] = {}
    with Timer(time_info, "main"):
        print(args)

        transformer = clean_fid_transformer if args.clean_fid else base_transformer

        generated_dataset = datasets[args.generated_data_type](
            root=args.generated_data_pth,
            transform=transformer,
            train=True,
            download=False
        )
        if isinstance(generated_dataset, Dataset):
            generated_data = DataLoader(generated_dataset, batch_size=50, num_workers=2)
        else:
            generated_data = generated_dataset

        real_data = None
        if args.real_data_pth is not None:
            real_dataset = datasets[args.real_data_type](
                root=args.real_data_pth,
                transform=transformer,
                train=False,
                download=False
            )
            if isinstance(real_dataset, Dataset):
                real_data = DataLoader(real_dataset, batch_size=50, num_workers=2)
            else:
                real_data = real_dataset

        metric = metrics[args.metric](
            real_data=real_data,
            use_torch=args.use_torch
        )

        result = metric.calc_metric(generated_data)
        print(result)

    print(f"required time: {time_info['duration/main']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate Inception Score or FID")

    parser.add_argument('--metric', type=str, required=True,
                        help='which metric to calculate (IS or FID)')

    parser.add_argument('--generated_data_type', type=str, required=False,
                        help='generated data type (ImageDataset, StandardDataset or stats')
    parser.add_argument('--generated_data_pth', type=str, required=False,
                        help='path to generated data (images or precalculated statistics)')

    parser.add_argument('--real_data_type', type=str, required=False,
                        help='real data type (ImageDataset, StandardDataset or stats)')
    parser.add_argument('--real_data_pth', type=str, required=False,
                        help='path to real data (images or precalculated statistics)')

    parser.add_argument('--clean_fid', action='store_true', required=False,
                        help='using clean fid preprocessing')

    parser.add_argument('--use_torch', action='store_true', required=False,
                        help='using pytorch as the matrix operations backend')

    run(parser.parse_args())
