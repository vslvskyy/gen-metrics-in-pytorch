import argparse
from typing import Dict, Callable, Union

from torch.utils.data import DataLoader, Dataset

from gan_metrics.base_class import metrics
from gan_metrics.utils import base_transformer, clean_fid_transformer, datasets, Timer


def get_data(
    data_type: str, data_pth: str, transformer: Callable, train: bool
) -> Union[DataLoader, str]:
    dataset = datasets[data_type](
        root=data_pth, transform=transformer, train=train, download=True
    )

    if isinstance(dataset, Dataset):
        return DataLoader(dataset, batch_size=50, num_workers=2)
    return dataset


def run(args: argparse.Namespace):
    time_info: Dict[str, float] = {}

    with Timer(time_info, "main"):
        print(args)

        transformer = clean_fid_transformer if args.clean_fid else base_transformer

        generated_data = get_data(
            args.generated_data_type, args.generated_data_pth, transformer, True
        )

        real_data = None
        if args.real_data_type is not None:
            real_data = get_data(
                args.real_data_type, args.real_data_pth, transformer, False
            )

        for metric_name in args.metric.strip().split(", "):
            metric = metrics[metric_name](real_data=real_data, use_torch=args.use_torch)

            metric_val, time_info = metric.calc_metric(generated_data)

            if args.time_info:
                print(metric_val, time_info)
            else:
                print(metric_val)

    if args.time_info:
        print(f"required time: {time_info['duration/main']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate Inception Score, FID or ImprovedPRD")

    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help="which metric to calculate (IS, FID, ImprovedPRD)",
    )

    parser.add_argument(
        "--generated_data_type",
        type=str,
        required=False,
        help="generated data type (ImageDataset, StandardDataset or stats)",
    )
    parser.add_argument(
        "--generated_data_pth",
        type=str,
        required=False,
        help="path to generated data (images or precalculated statistics)",
    )

    parser.add_argument(
        "--real_data_type",
        type=str,
        required=False,
        help="real data type (ImageDataset, StandardDataset or stats)",
    )
    parser.add_argument(
        "--real_data_pth",
        type=str,
        required=False,
        help="path to real data (images or precalculated statistics)",
    )

    parser.add_argument(
        "--clean_fid",
        action="store_true",
        required=False,
        help="using clean fid preprocessing",
    )

    parser.add_argument(
        "--use_torch",
        action="store_true",
        required=False,
        help="using pytorch as the matrix operations backend",
    )

    parser.add_argument(
        "--time_info",
        action="store_true",
        required=False,
        help="whether to display dict with required time information or not",
    )

    run(parser.parse_args())
