# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

from maro.cli.local.commands import run


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("conf_path", help="Path of the job deployment")
    parser.add_argument("--evaluate_only", action="store_true", help="Only run evaluation part of the workflow")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run(conf_path=args.conf_path, containerize=False, evaluate_only=args.evaluate_only)
