# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os

import yaml

"""Load from files"""


def load_cluster_details(cluster_name: str) -> dict:
    with open(os.path.expanduser(f"~/.maro/clusters/{cluster_name}/details.yml"), 'r') as fr:
        cluster_details = yaml.safe_load(fr)
    return cluster_details


def load_job_details(cluster_name: str, job_name: str) -> dict:
    with open(os.path.expanduser(f"~/.maro/clusters/{cluster_name}/jobs/{job_name}/details.yml"), 'r') as fr:
        job_details = yaml.safe_load(fr)
    return job_details
