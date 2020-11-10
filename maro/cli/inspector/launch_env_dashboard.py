import os


def launch_dashboard(ROOT_PATH, scenario):
    """Launch streamlit dashboard.

    Args:
        ROOT_PATH(str): Data folder path.
        scenario(str): Name of current scenario.
    """
    try:
        import streamlit
    except ImportError:
        os.system("pip install streamlit")
    streamlit.logger(f"{scenario} start")
    vis_path = os.path.expanduser("~/.maro/vis/template/visualization.py")
    os.system(rf"streamlit run {vis_path} -- {ROOT_PATH} {scenario}")
