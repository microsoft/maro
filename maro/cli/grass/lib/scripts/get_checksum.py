# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import os
import sys

from .utils import get_checksum

if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path')
    args = parser.parse_args()

    if os.path.exists(args.file_path):
        checksum = get_checksum(file_path=args.file_path)
    else:
        checksum = ""

    # Print job details
    sys.stdout.write(checksum)
