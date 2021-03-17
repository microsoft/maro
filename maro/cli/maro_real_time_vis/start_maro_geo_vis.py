# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import inspect
import os
import subprocess
import time

import requests

from maro.utils.exception.cli_exception import CliError
from maro.utils.logger import CliLogger

from .back_end.vis_app.data_process.request.request_params import request_settings

logger = CliLogger(name=__name__)


def start_geo_vis(start: str, experiment_name: str, front_end_port: int, **kwargs: dict):
    grader_path = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    if start == 'database':

        # Start the databse container.
        database_start_path = f"{grader_path}\\streamit\\server"
        subprocess.check_call(
            'sh run_docker.sh',
            cwd=database_start_path
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
        if front_end_port is not None:
            change_file_content(
                f"{exec_path}\\.env",
                "FRONT_END_PORT",
                f"FRONT_END_PORT={front_end_port}"
            )
            change_file_content(
                f"{exec_path}\\front_end\\nginx.conf",
                "listen",
                f"\t\tlisten\t\t{front_end_port};"
            )

        subprocess.check_call(
            'sh run_docker.sh',
            cwd=exec_path
        )
        back_end_path = f"{exec_path}\\back_end\\vis_app\\app.py"
        os.system(f"python {back_end_path}")

    else:
        raise CliError("Please input 'database' or 'service'.")


def change_file_content(file_path: str, key_words: str, dest_words: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            lines.append(line)
        f.close()
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in lines:
            if key_words in line:
                line = dest_words
                f.write('%s\n' % line)
            else:
                f.write('%s' % line)
