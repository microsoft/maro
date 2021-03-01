import inspect
import os
import subprocess


def start_geo_vis(**kwargs: dict):
    subprocess.check_call(
        'sh run_docker.sh',
        cwd=os.path.dirname(inspect.getfile(inspect.currentframe()))
    )
