import argparse
import fcntl
import json
import os
import platform
import pty
import select
import shlex
import struct
import subprocess
import termios
import time
import traceback

import pandas as pd
from flask import Flask, send_file, send_from_directory
from flask_socketio import SocketIO

port = 8080

app = Flask("Terminal-Service")
app.config["fd"] = None
app.config["pid"] = None
app.config["child_pid"] = None
app.config["cluster_list"] = []
app.config["cluster_status"] = {}
app.config["local_executor"] = {}
socketio = SocketIO(app)


class DashboardType():
    PROCESS = "process"
    LOCAL = "local"
    AZURE = "azure"
    ONPREMISES = "on-premises"


def set_terminal_size(fd, row, col, xpix=0, ypix=0):
    winsize = struct.pack("HHHH", row, col, xpix, ypix)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)


def read_and_forward_pty_output():
    max_read_bytes = 1024 * 20
    while True:
        socketio.sleep(0.01)
        if app.config["fd"]:
            timeout_sec = 0
            (data_ready, _, _) = select.select(
                [app.config["fd"]], [], [], timeout_sec)
            if data_ready:
                output = os.read(app.config["fd"], max_read_bytes).decode()
                socketio.emit(
                    "pty-output", {"output": output}, namespace="/pty")


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/<path:path>")
def root_folder(path: str):
    return send_from_directory(".", path)


@app.route("/assets/<path:path>")
def assets_files(path: str):
    return send_from_directory("assets", path)


@socketio.on("pty-input", namespace="/pty")
def pty_input(data):
    """write to the child pty. The pty sees this as if you are typing in a real
    terminal.
    """
    if app.config["fd"]:
        os.write(app.config["fd"], data["input"].encode())


@socketio.on("resize", namespace="/pty")
def resize(data):
    if app.config["fd"]:
        set_terminal_size(app.config["fd"], data["rows"], data["cols"])


@socketio.on("connect", namespace="/pty")
def connect():
    """new client connected"""

    if app.config["child_pid"]:
        # already started child process, don't start another
        return

    # create child process attached to a pty we can read from and write to
    (child_pid, fd) = pty.fork()
    if child_pid == 0:
        subprocess.run(app.config["cmd"])
    else:

        app.config["fd"] = fd
        app.config["child_pid"] = child_pid
        set_terminal_size(fd, 17, 75)
        cmd = " ".join(shlex.quote(c) for c in app.config["cmd"])
        print("child pid is", child_pid)
        print(
            f"starting background task with command `{cmd}` to continously read "
            "and forward pty output to client"
        )
        socketio.start_background_task(target=read_and_forward_pty_output)
        socketio.start_background_task(target=update_cluster_list)
        print("task started")


# Admin data support

def load_executor(cluster_name):
    if cluster_name == "process":
        from maro.cli.process.executor import ProcessExecutor
        executor = ProcessExecutor()
        cluster_type = DashboardType.PROCESS
    else:
        from maro.cli.utils.details_reader import DetailsReader
        cluster_details = DetailsReader.load_cluster_details(
            cluster_name=cluster_name)
        if cluster_details["mode"] == "grass/azure":
            from maro.cli.grass.executors.grass_azure_executor import GrassAzureExecutor
            executor = GrassAzureExecutor(cluster_name=cluster_name)
            cluster_type = DashboardType.AZURE
        elif cluster_details["mode"] == "grass/on-premises":
            from maro.cli.grass.executors.grass_on_premises_executor import GrassOnPremisesExecutor
            executor = GrassOnPremisesExecutor(cluster_name=cluster_name)
            cluster_type = DashboardType.ONPREMISES
        elif cluster_details["mode"] == "grass/local":
            from maro.cli.grass.executors.grass_local_executor import GrassLocalExecutor
            executor = GrassLocalExecutor(cluster_name=cluster_name)
            cluster_type = DashboardType.LOCAL
    return executor, cluster_type


def update_resource_dynamic(org_data, local_executor, dashboard_type):
    if dashboard_type != DashboardType.PROCESS:
        new_data = local_executor.get_resource_usage(0)
        for data_key in org_data.keys():
            org_data[data_key] = [new_data[data_key]]
    else:
        data_len = len(org_data['cpu'])
        new_data = local_executor.get_resource_usage(data_len)
        for data_key in org_data.keys():
            attr_data = new_data[data_key]
            data_array = []
            for data_byte in attr_data:
                data_str = data_byte.decode("utf-8")
                data_list = pd.Series(json.loads(data_str), dtype="float64")
                data_point = data_list.mean()
                if pd.isna(data_point):
                    data_point = 0
                if data_key == "memory":
                    data_point *= 100
                data_array.append(data_point)
            org_data[data_key].extend(data_array)


def update_cluster_list():
    while True:
        time.sleep(2)
        clusters = []
        from maro.cli.utils.params import GlobalPaths
        for root, _, files in os.walk(GlobalPaths.ABS_MARO_CLUSTERS, topdown=False):
            for name in files:
                if os.path.basename(name) == "cluster_details.yml":
                    clusters.append(os.path.basename(root))
        app.config["cluster_list"] = clusters

        clusters_removed = []
        for cluster_name in app.config["cluster_status"].keys():
            if cluster_name not in clusters:
                clusters_removed.append(cluster_name)
        for cluster_name in clusters_removed:
            del app.config["cluster_status"][cluster_name]

        for cluster_name in clusters:
            if cluster_name not in app.config["cluster_status"].keys():
                try:
                    app.config["cluster_status"][cluster_name] = {}
                    app.config["local_executor"][cluster_name],\
                        app.config["cluster_status"][cluster_name]["dashboard_type"] = load_executor(cluster_name)
                    app.config["cluster_status"][cluster_name]["cluster_name"] = cluster_name
                    local_executor = app.config["local_executor"][cluster_name]
                    app.config["cluster_status"][cluster_name]["resource_static"] = local_executor.get_resource()
                    app.config["cluster_status"][cluster_name]["resource_dynamic"] = {
                        "cpu": [],
                        "memory": [],
                        "gpu": []
                    }
                    update_resource_dynamic(app.config["cluster_status"][cluster_name]["resource_dynamic"],
                                            local_executor,
                                            app.config["cluster_status"][cluster_name]["dashboard_type"])
                    app.config["cluster_status"][cluster_name]["job_detail_data"] = local_executor.get_job_details()
                except Exception as e:
                    print(f"Failed to collect status for cluster {cluster_name}, error:{e}  {traceback.format_exc()}")
                    if cluster_name in app.config["cluster_status"].keys():
                        del app.config["cluster_status"][cluster_name]
            else:
                try:
                    local_executor = app.config["local_executor"][cluster_name]
                    update_resource_dynamic(app.config["cluster_status"][cluster_name]["resource_dynamic"],
                                            local_executor,
                                            app.config["cluster_status"][cluster_name]["dashboard_type"])
                    app.config["cluster_status"][cluster_name]["job_detail_data"] = local_executor.get_job_details()
                except Exception as e:
                    print(f"Failed to collect status for cluster {cluster_name}, error:{e}  {traceback.format_exc()}")
                    if cluster_name in app.config["cluster_status"].keys():
                        del app.config["cluster_status"][cluster_name]


@socketio.on("cluster_list", namespace="/pty")
def cluster_list():
    print("cluster list request received")
    socketio.emit("cluster_list", app.config["cluster_list"], namespace="/pty")


@socketio.on("cluster_status", namespace="/pty")
def cluster_status(data):
    print("cluster status request received")
    if data["cluster_name"] in app.config["cluster_status"].keys():
        socketio.emit("cluster_status", app.config["cluster_status"][data["cluster_name"]], namespace="/pty")
    else:
        socketio.emit("cluster_status", None, namespace="/pty")


def os_is_windows() -> bool:
    info = platform.platform()
    if "Windows" in info:
        return True
    else:
        return False


def shell_cmd() -> str:
    if os_is_windows():
        return "cmd.exe"
    else:
        return "bash"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Welcome to MARO. "
            "Repository: https://github.com/microsoft/maro"
            "Documentation: https://maro.readthedocs.io/en/latest/"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--port", default=port,
                        help="port to run server on")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="host to run server on (use 0.0.0.0 to allow access from other hosts)",
    )
    parser.add_argument("--debug", action="store_true",
                        help="debug the server")
    parser.add_argument("--version", action="store_true",
                        help="print version and exit")
    parser.add_argument(
        "--command", default=shell_cmd(), help="Command to run in the terminal"
    )
    parser.add_argument(
        "--cmd-args",
        default="",
        help="arguments to pass to command (i.e. --cmd-args='arg1 arg2 --flag')",
    )
    args = parser.parse_args()
    print(f"serving on http://127.0.0.1:{args.port}")
    app.config["cmd"] = [args.command] + shlex.split(args.cmd_args)
    socketio.run(app, debug=args.debug, port=args.port, host=args.host)


if __name__ == "__main__":
    main()
