# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import json

from redis import Redis

from .utils import load_cluster_details, load_job_details


def create_job_details(cluster_name: str, job_name: str):
    # Load details
    cluster_details = load_cluster_details(cluster_name=cluster_name)
    job_details = load_job_details(cluster_name=cluster_name, job_name=job_name)
    master_hostname = cluster_details['master']['hostname']
    redis_port = cluster_details['master']['redis']['port']

    # Add other parameters
    job_details['containers'] = {}

    redis = Redis(
        host=master_hostname,
        port=redis_port,
        charset="utf-8",
        decode_responses=True
    )
    redis.hset(
        f"{cluster_name}:job_details",
        job_name,
        json.dumps(job_details)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cluster_name')
    parser.add_argument('job_name')
    args = parser.parse_args()

    create_job_details(cluster_name=args.cluster_name, job_name=args.job_name)
