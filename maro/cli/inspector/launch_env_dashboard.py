import os


def launch_dashboard(ROOT_PATH, scenario):
    """Launch streamlit dashboard.

    Args:
        ROOT_PATH(str): Data folder path.
        scenario(str): Name of current scenario.
    """
    # os.system(rf'streamlit run ~\.maro\vis\template\visualization.py -- {ROOT_PATH} {scenario}')
    os.system(rf'streamlit run C:\Users\Administrator\source\repos\maro-vis\maro\cli\inspector\visualization.py -- {ROOT_PATH} {scenario}')
