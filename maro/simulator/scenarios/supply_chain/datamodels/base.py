

from abc import abstractmethod
from maro.backends.frame import NodeBase


class DataModelBase(NodeBase):
    @abstractmethod
    def initialize(self, configs):
        """Initialize the fields with configs, the config should be a dict."""
        pass

    @abstractmethod
    def reset(self):
        """Reset after each episode"""
        pass
