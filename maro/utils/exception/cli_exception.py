# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .base_exception import MAROException


class CliException(MAROException):
    """The general CLI Exception."""

    def __init__(self, message: str = None, error_code: int = 3000):
        super().__init__(error_code, message)

    def get_message(self) -> str:
        """Get the error message of the Exception.

        Returns:
            str: Error message.
        """
        return self.strerror


class CommandError(CliException):
    """Failed execution error of CLI command."""

    def __init__(self, cli_command: str, message: str = None):
        super().__init__(error_code=3001, message=message)
        self.cli_command = cli_command

    def __str__(self):
        return f"command: {self.cli_command}\n {self.strerror}"


class ParsingError(CliException):
    """Parsing error."""

    def __init__(self, message: str = None):
        super().__init__(error_code=3002, message=message)


class DeploymentError(CliException):
    """Failed deployment error."""

    def __init__(self, message: str = None):
        super().__init__(error_code=3003, message=message)
