# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import hashlib
import os
import sys


def get_checksum(file_path: str, block_size=128):
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(block_size * md5.block_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


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
