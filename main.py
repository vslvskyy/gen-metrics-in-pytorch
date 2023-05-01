import argparse
from typing import Dict

from gan_metrics.base_class import metrics
from gan_metrics.utils import Timer


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate Inception Score or FID")

    parser.add_argument("--metric", choices=metrics.items.keys(), required=True,
                        help=f"which metric to calculate ({', '.join(metrics.items.keys())})")

    parser.add_argument("--gen_type", type=str, required=False,
                        help="generated data type (folder, some standar dataset or stats)")
    parser.add_argument("--gen_path", type=str, required=True,
                        help="path to generated data (images or precalculated statistics)")

    parser.add_argument("--real_type", type=str, required=False,
                        help="real data type (folder, some standar dataset or stats)")
    parser.add_argument("--real_path", type=str, required=False,
                        help="path to real data (images or precalculated statistics)")

    parser.add_argument("--gen_save_path", type=str, required=False,
                        help="path to save generated data features or statistics")
    parser.add_argument("--real_save_path", type=str, required=False,
                        help="path to save real data features or statistics")

    args = parser.parse_args()
    time_info: Dict[str, float] = {}
    with Timer(time_info, "main"):
        metric = metrics[args.metric]()
        result = metric(
            gen_path=args.gen_path, gen_type=args.gen_type,
            real_path=args.real_path, real_type=args.real_type,
            gen_save_path=args.gen_save_path, real_save_path=args.real_save_path
    )
    print(result)
    print(f"required time: {time_info['duration/main']}")
