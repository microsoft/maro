# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse

from redis import Redis

from .utils import delete_node_details, get_node_details, load_cluster_details, set_node_details

if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument('cluster_name')
    parser.add_argument('node_name')
    parser.add_argument('action')
    args = parser.parse_args()

    # Load details
    cluster_details = load_cluster_details(cluster_name=args.cluster_name)
    master_hostname = cluster_details['master']['hostname']
    redis_port = cluster_details['master']['redis']['port']
    redis = Redis(
        host=master_hostname,
        port=redis_port,
        charset="utf-8", decode_responses=True
    )

    if args.action == 'create':
        node_details = get_node_details(
            redis,
            cluster_name=args.cluster_name,
            node_name=args.node_name
        )
        node_details['image_files'] = {}
        node_details['containers'] = {}
        node_details['state'] = 'Running'
        set_node_details(
            redis=redis,
            cluster_name=args.cluster_name,
            node_name=args.node_name,
            node_details=node_details
        )
    elif args.action == 'delete':
        delete_node_details(
            redis=redis,
            cluster_name=args.cluster_name,
            node_name=args.node_name,
        )
    elif args.action == 'stop':
        node_details = get_node_details(
            redis=redis,
            cluster_name=args.cluster_name,
            node_name=args.node_name
        )
        node_details['state'] = 'Stopped'
        set_node_details(
            redis=redis,
            cluster_name=args.cluster_name,
            node_name=args.node_name,
            node_details=node_details
        )
    elif args.action == 'start':
        node_details = get_node_details(
            redis=redis,
            cluster_name=args.cluster_name,
            node_name=args.node_name
        )
        node_details['state'] = 'Running'
        set_node_details(
            redis=redis,
            cluster_name=args.cluster_name,
            node_name=args.node_name,
            node_details=node_details
        )
