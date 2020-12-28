# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse

from redis import Redis

from ..utils.details import del_master_details, load_cluster_details

if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument("cluster_name")
    args = parser.parse_args()

    # Load details
    cluster_details = load_cluster_details(cluster_name=args.cluster_name)
    master_hostname = cluster_details["master"]["hostname"]
    redis_port = cluster_details["master"]["redis"]["port"]

    # Get nodes details
    redis = Redis(
        host=master_hostname,
        port=redis_port,
        charset="utf-8", decode_responses=True
    )
    del_master_details(
        redis=redis,
        cluster_name=args.cluster_name
    )
