# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import json
import sys

from redis import Redis

from .utils import get_master_details, load_cluster_details

if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument('cluster_name')
    args = parser.parse_args()

    # Load details
    cluster_details = load_cluster_details(cluster_name=args.cluster_name)
    master_hostname = cluster_details['master']['hostname']
    redis_port = cluster_details['master']['redis']['port']

    # Get nodes details
    redis = Redis(
        host=master_hostname,
        port=redis_port,
        charset="utf-8", decode_responses=True
    )
    master_details = get_master_details(
        redis=redis,
        cluster_name=args.cluster_name
    )

    # Print job details
    sys.stdout.write(json.dumps(master_details))
