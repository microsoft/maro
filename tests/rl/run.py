# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

from maro.cli.local.commands import run
from maro.utils.utils import set_seeds


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("conf_path", help="Path of the job deployment")
    parser.add_argument("--evaluate_only", action="store_true", help="Only run evaluation part of the workflow")
    parser.add_argument("--seed", type=int, help="The random seed set before running this job")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.seed is not None:
        set_seeds(seed=args.seed)
    run(conf_path=args.conf_path, containerize=False, evaluate_only=args.evaluate_only)
