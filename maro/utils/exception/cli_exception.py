# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .base_exception import MAROException

# First Layer.


class CliError(MAROException):
    """ Base class for all MARO CLI errors."""

    def __init__(self, message: str = None, error_code: int = 3000):
        super().__init__(error_code, message)

    def get_message(self) -> str:
        """ Get the error message of the Exception.

        Returns:
            str: Error message.
        """
        return self.strerror


# Second Layer.


class UserFault(CliError):
    """ Users should be responsible for the errors.
    ErrorCode with 30XX."""
    pass


class ClientError(CliError):
    """ MARO CLI should be responsible for the errors.
    ErrorCode with 31XX."""
    pass


class ServiceError(CliError):
    """ MARO Services should be responsible for the errors.
    ErrorCode with 32XX."""
    pass


# Third Layer.


class CommandNotFoundError(UserFault):
    """ Command is misspelled or not recognized by MARO CLI."""

    def __init__(self, message: str = None, usage: str = ""):
        self.usage = usage
        super().__init__(error_code=3000, message=message)


class BadRequestError(UserFault):
    """ Bad request from client."""

    def __init__(self, message: str = None):
        super().__init__(error_code=3001, message=message)


class InvalidDeploymentTemplateError(UserFault):
    """ MARO deployment template validation fails."""

    def __init__(self, message: str = None):
        super().__init__(error_code=3002, message=message)


class DeploymentError(UserFault):
    """ MARO deployment fails."""

    def __init__(self, message: str = None):
        super().__init__(error_code=3003, message=message)


class FileOperationError(UserFault):
    """ For file or directory operation related errors. """

    def __init__(self, message: str = None):
        super().__init__(error_code=3004, message=message)


class CliInternalError(ClientError):
    """ MARO CLI internal error."""

    def __init__(self, message: str = None):
        super().__init__(error_code=3100, message=message)


class ClusterInternalError(ServiceError):
    """ MARO Cluster internal error."""

    def __init__(self, message: str = None):
        super().__init__(error_code=3200, message=message)


class CommandExecutionError(ServiceError):
    """ Subprocess execution error."""

    def __init__(self, message: str = None, command: str = None):
        self.command = command
        super().__init__(error_code=3201, message=message)

    def get_message(self) -> str:
        """ Get the error message of the Exception.

        Returns:
            str: Error message.
        """
        return f"Command: {self.command}\nErrorMessage: {self.strerror}"


class CommandError(CliError):
    """ Failed execution error of CLI command."""

    def __init__(self, cli_command: str, message: str = None):
        super().__init__(error_code=3001, message=message)
        self.cli_command = cli_command

    def __str__(self):
        return f"command: {self.cli_command}\n {self.strerror}"


class ProcessInternalError(UserFault):
    """ Errors in MARO CLI process mode. """

    def __init__(self, message: str = None):
        super().__init__(error_code=3005, message=message)
