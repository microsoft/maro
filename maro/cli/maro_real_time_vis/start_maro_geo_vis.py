import os
import inspect
import subprocess


def start_geo_vis(**kwargs: dict):
    print("test")
    subprocess.check_call(
        'sh run_docker.sh',
        cwd=os.path.dirname(inspect.getfile(inspect.currentframe()))
    )

