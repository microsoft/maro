# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
import os

from redis import Redis
from typing import Any

from ..cli.utils.details_reader import DetailsReader


def report_final_result(result: Any):
    # FIXME
    # local_master_details = DetailsReader.load_local_master_details()
    cluster_name = os.environ['CLUSTER_NAME']
    local_cluster_details = DetailsReader.load_cluster_details(cluster_name)
    redis_connection = Redis(host='127.0.0.1', port=local_cluster_details['master']['redis']['port'],
                             charset='utf-8', decode_responses=True)
    name = os.environ['FINAL_RESULT_KEY']
    job_name = os.environ['JOB_NAME']
    redis_connection.hset(name, job_name, json.dumps(result))
