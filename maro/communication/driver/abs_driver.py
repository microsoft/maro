# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# native lib
from abc import ABC, abstractmethod
from typing import Dict


class AbsDriver(ABC):
    """Abstract class of the communication driver."""

    @property
    @abstractmethod
    def address(self):
        """Return the socket's address.

        Returns:
            Based on the real socket driver, the socket's address usually be a Dict with, \n
                the key of dict is socket's type, \n
                the value of dict is socket's address; usually, the format is protocol+ip+port.
        """
        pass

    @abstractmethod
    def connect(self, peers_address: Dict):
        """Build the connection with other peers which is given by the peer address.
        
        Args:
            peers_address (Dict): Peers socket address dict, \n
                the key of dict is the peer's name, \n
                the value of dict is the peer's socket connection address.
        """
        pass

    @abstractmethod
    def receive(self):
        """Receive message."""
        pass

    @abstractmethod
    def send(self, message):
        """Unicast send message."""
        pass

    @abstractmethod
    def broadcast(self, message):
        """Broadcast send message."""
        pass
