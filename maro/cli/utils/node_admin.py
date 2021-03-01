# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

MONITOR_PATH = os.path.expanduser("~/.maro/web_terminal/monitor.py")
TERMINAL_PATH = os.path.expanduser("~/.maro/web_terminal/terminal-srv.py")

def start_admin(*args, **kwargs):
    os.system(
        f"streamlit run {MONITOR_PATH} & "
    )

    os.system(
        f"cd {os.path.dirname(TERMINAL_PATH)} ;"
        f"python {TERMINAL_PATH} &"
    )


def stop_admin(*args, **kwargs):
    os.system(
        f"pkill -f '{MONITOR_PATH}' "
    )

    os.system(
        f"pkill -f '{TERMINAL_PATH}' "
    )
