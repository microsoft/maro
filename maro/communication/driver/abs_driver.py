# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# native lib
from abc import ABC, abstractmethod


class AbsDriver(ABC):
    """Abstract class of the communication driver."""

    @property
    @abstractmethod
    def address(self):
        """Dict: The socket's address. Based on the real socket driver, it usually be a ``Dict``."""
        raise NotImplementedError

    @abstractmethod
    def connect(self, peers_address):
        """Build the connection with other peers which is given by the peer address.

        Args:
            peers_address: The store of peers' socket address. Based on the real socket driver,
                the peers' socket address usually be a ``Dict``.
        """
        raise NotImplementedError

    @abstractmethod
    def receive(self):
        """Receive message."""
        raise NotImplementedError

    @abstractmethod
    def send(self, message):
        """Unicast send message."""
        raise NotImplementedError

    @abstractmethod
    def broadcast(self, component_type, message):
        """Broadcast send message."""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Close all sockets."""
        raise NotImplementedError
