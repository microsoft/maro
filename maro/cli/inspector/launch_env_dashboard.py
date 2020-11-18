import os


def launch_dashboard(source_path: str, scenario: enumerate, epoch_num: int, prefix: str):
    """Launch streamlit dashboard.

    Args:
        source_path (str): Data folder path.
        scenario (enumerate): Name of current scenario.
        epoch_num (int): Number of epochs.
        prefix (str): Prefix of data folders.
    """
    vis_path = os.path.expanduser("~/.maro/vis/templates/visualization.py")
    os.system(
        f"streamlit run {vis_path} "
        f"-- --source_path {source_path} --scenario {scenario.value} --epoch_num {epoch_num} --prefix {prefix}")
