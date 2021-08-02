# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import inspect
import os
import subprocess
import time

import requests

from maro.utils.exception.cli_exception import CliError

from .back_end.vis_app.data_process.request.request_params import request_settings

GEO_FRONT_SERVICE_DOCKER_IMAGE = "maro2020/geo_front_service"
DEFAULT_FRONT_END_PORT = 8080


def start_geo_vis(start: str, experiment_name: str, front_end_port: int, **kwargs: dict):
    grader_path = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    if start == 'database':

        # Start the database container.
        database_start_path = f"{grader_path}/streamit/server"
        subprocess.check_call(
            'sh run_docker.sh',
            cwd=database_start_path,
            shell=True
        )
    elif start == 'service':
        if experiment_name is None:
            raise CliError("Please input experiment name.")
        find_exp_name_params = {
            "query": f"select * from maro.experiments where name='{experiment_name}'",
            "count": "true"
        }
        find_exp_name = requests.get(
            url=request_settings.request_url.value,
            headers=request_settings.request_header.value,
            params=find_exp_name_params
        ).json()
        if find_exp_name["dataset"] == []:
            raise CliError("Please input a valid experiment name.")
        # Create experiment display list table.
        no_table_error = False
        params = {
            "query": "select * from pending_experiments",
            "count": "true"
        }
        try:
            requests.get(
                url=request_settings.request_url.value,
                headers=request_settings.request_header.value,
                params=params
            ).json()
        except ConnectionError:
            no_table_error = True
        else:
            no_table_error = True

        if no_table_error:
            create_params = {
                "query": "Create table pending_experiments(name STRING, time LONG)",
            }
            requests.get(
                url=request_settings.request_url.value,
                headers=request_settings.request_header.value,
                params=create_params
            ).json()

        current_time = int(time.time())
        next_exp_params = {
            "query": f"INSERT INTO pending_experiments(name, time) VALUES('{experiment_name}', {current_time})",
        }
        requests.get(
            url=request_settings.request_url.value,
            headers=request_settings.request_header.value,
            params=next_exp_params
        ).json()

        # Start front-end docker container.
        exec_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
        os.system(f"docker pull {GEO_FRONT_SERVICE_DOCKER_IMAGE}")
        os.system("docker stop geo-vis")
        os.system("docker rm geo-vis")
        os.system(
            f"docker run -d -p {front_end_port if front_end_port is not None else DEFAULT_FRONT_END_PORT}:8080 "
            f"--name geo-vis {GEO_FRONT_SERVICE_DOCKER_IMAGE}"
        )
        back_end_path = f"{exec_path}/back_end/vis_app/app.py"
        os.system(f"python {back_end_path}")
    else:
        raise CliError("Please input 'database' or 'service'.")
