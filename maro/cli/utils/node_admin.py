# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os


def start_admin(*args, **kwargs):
    monitor_path = os.path.expanduser("~/.maro/web_terminal/monitor.py")
    os.system(
        f"streamlit run {monitor_path} & "
    )

    terminal_path = os.path.expanduser("~/.maro/web_terminal/terminal-srv.py")
    os.system(
        f"cd {os.path.dirname(terminal_path)} ;"
        f"python {terminal_path} &"
    )


def stop_admin(*args, **kwargs):
    monitor_path = os.path.expanduser("~/.maro/web_terminal/monitor.py")
    os.system(
        f"pkill -f '{monitor_path}' "
    )

    terminal_path = os.path.expanduser("~/.maro/web_terminal/terminal-srv.py")
    os.system(
        f"pkill -f '{terminal_path}' "
    )
