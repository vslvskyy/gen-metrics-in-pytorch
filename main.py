import argparse
from typing import Dict

from gen_metrics.base_class import metrics
from gen_metrics.datasets import datasets
from gen_metrics.utils import Timer


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate Generative Model Metric")

    parser.add_argument("--metric", choices=metrics.items.keys(), required=True,
                        help=f"which metric to calculate ({', '.join(metrics.items.keys())})")

    parser.add_argument("--gen_data", type=str, nargs=2, required=True,
                        help=f"generated data type {list(datasets.items.keys()) + ['stats']} and path to images")
    parser.add_argument("--real_data", type=str, nargs=2, required=False, default=(None, None),
                        help=f"real data type {list(datasets.items.keys()) + ['stats']} and path to images")

    parser.add_argument("--gen_save_path", type=str, required=False,
                        help="path to save generated data features or statistics")
    parser.add_argument("--real_save_path", type=str, required=False,
                        help="path to save real data features or statistics")

    args = parser.parse_args()

    time_info: Dict[str, float] = {}
    with Timer(time_info, "main"):
        metric = metrics[args.metric]()
        result = metric(
            gen_path=args.gen_data[1], real_path=args.real_data[1],
            gen_type=args.gen_data[0], real_type=args.real_data[0],
            gen_save_path=args.gen_save_path, real_save_path=args.real_save_path
    )

    print(result)
    print(f"required time: {time_info['duration/main']}")
