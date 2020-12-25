# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import platform
import subprocess
import sys


class SubProcess:
    @staticmethod
    def run(command: str, timeout: int = None) -> str:
        """Run one-time command with subprocess.run().

        Args:
            command (str): command to be executed.
            timeout (int): timeout in seconds.

        Returns:
            str: return stdout of the command.
        """
        if platform.system() == "Windows":
            command = f"powershell.exe -Command \"{command}\""
        completed_process = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout
        )
        if completed_process.returncode != 0:
            raise Exception(completed_process.stderr)
        return completed_process.stdout

    @staticmethod
    def interactive_run(command: str) -> None:
        """Run one-time command with subprocess.popen() and write stdout output interactively.

        Args:
            command (str): command to be executed.

        Returns:
            None.
        """
        if platform.system() == "Windows":
            command = "powershell.exe " + command
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        while True:
            next_line = process.stdout.readline()
            if next_line == "" and process.poll() is not None:
                break
            sys.stdout.write(next_line)
            sys.stdout.flush()
        stdout, stderr = process.communicate()
        if stderr:
            sys.stderr.write(stderr.strip("\n"))
