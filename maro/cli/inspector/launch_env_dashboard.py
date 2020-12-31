# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from .params import GlobalScenarios


def launch_dashboard(source_path: str, scenario: GlobalScenarios, epoch_num: int, prefix: str):
    """Launch streamlit dashboard.

    Args:
        source_path (str): The root path of the dumped snapshots data for the corresponding experiment.
        scenario (GlobalScenarios): Name of current scenario.
        epoch_num (int): Number of epochs.
        prefix (str): Prefix of data folders.
    """
    vis_path = os.path.expanduser("~/.maro/vis/templates/visualization.py")
    os.system(
        f"streamlit run {vis_path} "
        f"-- --source_path {source_path} --scenario {scenario.value} --epoch_num {epoch_num} --prefix {prefix}"
    )
