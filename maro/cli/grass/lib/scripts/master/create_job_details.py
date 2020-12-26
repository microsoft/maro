# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse

from redis import Redis

from .utils.details import load_cluster_details, load_job_details, set_job_details

if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument("cluster_name")
    parser.add_argument("job_name")
    args = parser.parse_args()

    # Load details
    cluster_details = load_cluster_details(cluster_name=args.cluster_name)
    job_details = load_job_details(cluster_name=args.cluster_name, job_name=args.job_name)
    master_hostname = cluster_details["master"]["hostname"]
    redis_port = cluster_details["master"]["redis"]["port"]

    # Add other parameters
    job_details["containers"] = {}

    # Create job details
    redis = Redis(
        host=master_hostname,
        port=redis_port,
        charset="utf-8",
        decode_responses=True
    )
    set_job_details(
        redis=redis,
        cluster_name=args.cluster_name,
        job_name=args.job_name,
        job_details=job_details
    )
