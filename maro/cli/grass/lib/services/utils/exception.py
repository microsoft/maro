# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


# First Layer.


class AgentError(Exception):
    """Base error class for all MARO Grass Agents."""


# Second Layer.


class UserFault(AgentError):
    """Users should be responsible for the errors."""


class ServiceError(AgentError):
    """MARO Services should be responsible for the errors."""


# Third Layer.


class ResourceAllocationFailed(UserFault):
    """Resources are insufficient, unable to allocate."""


class StartContainerError(ServiceError):
    """Error when starting containers."""


class CommandExecutionError(ServiceError):
    """Failed to execute shell commands."""


class ConnectionFailed(ServiceError):
    """Failed to connect to other nodes."""
