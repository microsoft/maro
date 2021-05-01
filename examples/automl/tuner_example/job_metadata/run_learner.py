import time

import json
import os

from redis import Redis
from typing import Any

def report_final_result(result: Any):
    host = os.environ['REDIS_HOST']
    port = os.environ['REDIS_PORT']
    name = os.environ['FINAL_RESULT_KEY']
    job_name = os.environ['JOB_NAME']
    redis_connection = Redis(host=host, port=port, charset='utf-8', decode_responses=True)
    redis_connection.hset(name, job_name, json.dumps(result))

time.sleep(5)
report_final_result(0.98)
