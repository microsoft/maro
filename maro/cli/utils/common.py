# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import subprocess
import sys
from collections import deque

import psutil

from maro.utils import Logger


def close_by_pid(pid: int, recursive: bool = True):
    if not psutil.pid_exists(pid):
        return

    proc = psutil.Process(pid)
    if recursive:
        for child in proc.children(recursive=recursive):
            child.kill()

    proc.kill()


def get_child_pids(parent_pid):
    # command = f"ps -o pid --ppid {parent_pid} --noheaders"
    # get_children_pid_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    # children_pids = get_children_pid_process.stdout.read()
    # get_children_pid_process.wait(timeout=2)

    # # Convert into list or int
    # try:
    #     children_pids = int(children_pids)
    # except ValueError:
    #     children_pids = children_pids.decode().split("\n")
    #     children_pids = [int(pid) for pid in children_pids[:-1]]

    # return children_pids
    try:
        return [child.pid for child in psutil.Process(parent_pid).children(recursive=True)]
    except psutil.NoSuchProcess:
        print(f"No process with PID {parent_pid} found")
        return


def get_redis_pid_by_port(port: int):
    get_redis_pid_command = f"pidof 'redis-server *:{port}'"
    get_redis_pid_process = subprocess.Popen(get_redis_pid_command, shell=True, stdout=subprocess.PIPE)
    redis_pid = int(get_redis_pid_process.stdout.read())
    get_redis_pid_process.wait()
    return redis_pid


def exit(state: int = 0, msg: str = None):
    """Exit and show msg in sys.stderr"""
    if msg is not None:
        sys.stderr.write(msg)

    sys.exit(state)


def get_last_k_lines(file_name: str, k: int):
    """
    Helper function to retrieve the last K lines from a file in a memory-efficient way.

    Code slightly adapted from https://thispointer.com/python-get-last-n-lines-of-a-text-file-like-tail-command/
    """
    # Create an empty list to keep the track of last k lines
    lines = deque()
    # Open file for reading in binary mode
    with open(file_name, "rb") as fp:
        # Move the cursor to the end of the file
        fp.seek(0, os.SEEK_END)
        # Create a buffer to keep the last read line
        buffer = bytearray()
        # Get the current position of pointer i.e eof
        ptr = fp.tell()
        # Loop till pointer reaches the top of the file
        while ptr >= 0:
            # Move the file pointer to the location pointed by ptr
            fp.seek(ptr)
            # Shift pointer location by -1
            ptr -= 1
            # read that byte / character
            new_byte = fp.read(1)
            # If the read byte is new line character then it means one line is read
            if new_byte != b"\n":
                # If last read character is not eol then add it in buffer
                buffer.extend(new_byte)
            elif buffer:
                lines.appendleft(buffer.decode()[::-1])
                if len(lines) == k:
                    return lines
                # Reinitialize the byte array to save next line
                buffer.clear()

        # As file is read completely, if there is still data in buffer, then it's the first of the last K lines.
        if buffer:
            lines.appendleft(buffer.decode()[::-1])

    return lines


def show_log(log_path: str, tail: int = -1, logger: Logger = None):
    print_fn = logger.info if logger else print
    if tail == -1:
        with open(log_path, "r") as fp:
            for line in fp:
                print_fn(line.rstrip("\n"))
    else:
        for line in get_last_k_lines(log_path, tail):
            print_fn(line)


def format_env_vars(env: dict, mode: str = "proc"):
    if mode == "proc":
        return env

    if mode == "docker":
        env_opt_list = []
        for key, val in env.items():
            env_opt_list.extend(["--env", f"{key}={val}"])
        return env_opt_list

    if mode == "docker-compose":
        return [f"{key}={val}" for key, val in env.items()]

    if mode == "k8s":
        return [{"name": key, "value": val} for key, val in env.items()]

    raise ValueError(f"'mode' should be one of 'proc', 'docker', 'docker-compose', 'k8s', got {mode}")
