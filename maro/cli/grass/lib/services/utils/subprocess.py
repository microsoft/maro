# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import subprocess

from .exception import CommandExecutionError


class Subprocess:
    """Wrapper class of subprocess
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
        # TODO: Windows node
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
            raise CommandExecutionError(completed_process.stderr)
        return completed_process.stdout.strip("\n")
