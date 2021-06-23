# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod


class AbsEarlyStopper(ABC):
    def __init__(self):
        super().__init__()
        self.metric_history = []

    def push(self, metric):
        self.metric_history.append(metric)

    @abstractmethod
    def stop(self) -> bool:
        raise NotImplementedError
