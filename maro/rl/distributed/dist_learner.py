# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import List, Union

import numpy as np

from maro.communication import SessionMessage, SessionType

from .base_dist_learner import BaseDistLearner
from .common import MessageTag, PayloadKey


class DistLearner(BaseDistLearner):
    """Distributed learner that broadcasts models to remote actors."""
    def collect(self):
        ep = self._scheduler.iter
        unfinished = set(self._actors)
        for msg in self._proxy.receive():
            if msg.tag == MessageTag.FINISHED:
                if msg.payload[PayloadKey.EPISODE] != ep:
                    self._logger.info(
                        f"Ignore a message of {msg.tag} with ep {msg.payload[PayloadKey.EPISODE]} (current ep: {ep})"
                    )
                    continue
                unfinished.discard(msg.source)
                # If enough update messages have been received, call _update() and break out of the loop to start the
                # next episode.
                result = self._registry_table.push(msg)
                if result:
                    performance, details = result[0]
                    break
       # Send a TERMINATE_EPISODE cmd to unfinished actors to catch them up.
        if unfinished:
            self.terminate_rollout(list(unfinished))

        return performance, details
