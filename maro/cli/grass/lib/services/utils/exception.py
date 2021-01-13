# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


# First Layer.

class AgentError(Exception):
    """ Base error class for all MARO Grass Agents."""
    pass


# Second Layer.


class UserFault(AgentError):
    """ Users should be responsible for the errors."""
    pass


class ServiceError(AgentError):
    """ MARO Services should be responsible for the errors."""
    pass


# Third Layer.

class ResourceAllocationFailed(UserFault):
    """ Resources are insufficient, unable to allocate."""
    pass


class StartContainerError(ServiceError):
    """ Error when starting containers."""
    pass


class CommandExecutionError(ServiceError):
    """ Failed to execute shell commands."""
    pass


class ConnectionFailed(ServiceError):
    """ Failed to connect to other nodes."""
    pass
