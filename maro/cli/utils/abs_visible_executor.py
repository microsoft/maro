# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# native lib
from abc import ABC, abstractmethod


class AbsVisibleExecutor(ABC):
    """Abstract class of the visible executor."""

    @abstractmethod
    def get_job_details(self):
        """Get job details."""
        pass

    @abstractmethod
    def get_job_queue(self):
        """Get pending job and killed job queue."""
        pass

    @abstractmethod
    def get_resource(self):
        """Get cluster resource."""
        pass

    @abstractmethod
    def get_resource_usage(self, previous_length: int):
        """Get cluster resource usage."""
        pass
