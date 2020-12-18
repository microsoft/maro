# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse

from redis import Redis

from .utils import load_cluster_details


def create_killed_job_ticket(cluster_name: str, job_name: str):
    # Load details
    cluster_details = load_cluster_details(cluster_name=cluster_name)
    master_hostname = cluster_details['master']['hostname']
    redis_port = cluster_details['master']['redis']['port']

    redis = Redis(
        host=master_hostname,
        port=redis_port,
        charset="utf-8",
        decode_responses=True
    )
    redis.lpush(
        f"{cluster_name}:killed_job_tickets",
        job_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cluster_name')
    parser.add_argument('job_name')
    args = parser.parse_args()

    create_killed_job_ticket(
        cluster_name=args.cluster_name, job_name=args.job_name)
