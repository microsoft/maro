# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import base64
import json

from redis import Redis

from .utils import load_cluster_details, set_node_details

if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument('cluster_name')
    parser.add_argument('node_name')
    parser.add_argument('node_details')
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
    set_node_details(
        redis=redis,
        cluster_name=args.cluster_name,
        node_name=args.node_name,
        node_details=json.loads(base64.b64decode(args.node_details).decode('utf8'))
    )
