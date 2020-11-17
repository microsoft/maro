import os


def launch_dashboard(source_path: str, scenario: enumerate, epoch_num: int, prefix: str):
    """Launch streamlit dashboard.

    Args:
        source_path (str): Data folder path.
        scenario (enumerate): Name of current scenario.
    """
    #vis_path = os.path.expanduser("~/.maro/vis/template/visualization.py")
    vis_path = r"C:\Users\Administrator\source\repos\maro-vis\maro\cli\inspector\visualization.py"
    os.system(f"streamlit run {vis_path} "
              f"-- --source_path {source_path} --scenario {scenario.value} --epoch_num {epoch_num} --prefix {prefix}")
