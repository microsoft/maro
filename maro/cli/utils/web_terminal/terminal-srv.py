import argparse
import fcntl
import os
import platform
import pty
import select
import shlex
import struct
import subprocess
import termios

from flask import Flask, redirect, send_file, send_from_directory
from flask_socketio import SocketIO

port = 8080

app = Flask("Terminal-Service")
app.config["fd"] = None
app.config["pid"] = None
app.config["child_pid"] = None
socketio = SocketIO(app)


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


@app.route("/dashboard")
def dashboard():
    return redirect('http://localhost:8501', code=301)


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
        set_terminal_size(fd, 20, 50)
        cmd = " ".join(shlex.quote(c) for c in app.config["cmd"])
        print("child pid is", child_pid)
        print(
            f"starting background task with command `{cmd}` to continously read "
            "and forward pty output to client"
        )
        socketio.start_background_task(target=read_and_forward_pty_output)
        print("task started")


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
