# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import subprocess
import sys


class Subprocess:
    """Wrapper class of subprocess, with CliException integrated.
    """

    @staticmethod
    def run(command: str, timeout: int = None) -> str:
        """Run one-time command with subprocess.run().

        Args:
            command (str): command to be executed.
            timeout (int): timeout in seconds.

        Returns:
            str: return stdout of the command.
        """
        # TODO: Windows master
        completed_process = subprocess.run(
            command,
            executable="/bin/bash",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout
        )
        if completed_process.returncode != 0:
            raise Exception(completed_process.stderr)
        return completed_process.stdout.strip("\n")

    @staticmethod
    def interactive_run(command: str) -> None:
        """Run one-time command with subprocess.popen() and write stdout output interactively.

        Args:
            command (str): command to be executed.

        Returns:
            None.
        """
        # TODO: Windows master
        process = subprocess.Popen(
            command,
            executable="/bin/bash",
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
        _, stderr = process.communicate()
        if stderr:
            sys.stderr.write(stderr)
