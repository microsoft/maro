# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from multiprocessing import Queue

import numpy
import torch

from .common import MessageType
from .sender import StreamitSender


def ensure_state(func):
    """Decorator that used to make sure sender process already started or not paused,
        or ignore current function call."""

    def _wrapper(*args, **kwargs):
        client_instance: StreamitClient = args[0]

        if not client_instance._is_started or client_instance._is_paused:
            return
        else:
            return func(*args, **kwargs)

    return _wrapper


class StreamitClient:
    """Client that used to collect data and stream the to server.

    Args:
        experiment_name (str): Name of current experiment, must be unique.
        host (str): Host ip of data service.
    """

    def __init__(self, experiment_name: str, host="127.0.0.1"):
        self._sender: StreamitSender
        self._data_queue = Queue()

        self._cur_episode = 0
        self._cur_tick = 0

        self._experiment_name = experiment_name
        self._host = host

        self._is_started = False
        self._is_paused = False

    @ensure_state
    def pause(self, is_parse=True):
        """Pause data collecting, will ignore following data.

        Args:
            is_parse (bool): Is stop collecting? True to stop, False to accept data. Default is True.
        """
        self._is_paused = is_parse

    @ensure_state
    def info(self, scenario: str, topology: str, durations: int, **kwargs):
        """Send information about current experiment, used to store into 'maro.experiments' table.

        Args:
            scenario (str): Scenario name of current experiment.
            topology (str): Topology name of current scenario.
            durations (int): Durations of each episode.
            kwargs (dict): Additional information to same.
        """
        # TODO: maybe it is better make it as parmater of with statement, so we can accept more info.
        self._put(MessageType.Experiment, (scenario, topology, durations, kwargs))

    @ensure_state
    def tick(self, tick: int):
        """Update current tick.

        Args:
            tick (int): Current tick.
            """
        self._cur_tick = tick

        self._put(MessageType.Tick, tick)

    @ensure_state
    def episode(self, episode: int):
        """Update current episode.

        Args:
            episode (int): Current episode.
        """
        self._cur_episode = episode

        self._put(MessageType.Episode, episode)

    @ensure_state
    def data(self, category: str, **kwargs):
        """Send data for specified category.

        Examples:
            streamit.data("my_category", name="user_name", age=10)

        Args:
            category (str): Category name of current data collection.
            kwargs (dict): Named data to send of current category.
        """
        self._put(MessageType.Data, (category, kwargs))

    @ensure_state
    def complex(self, category: str, value: dict):
        """This method will split value dictionary into small tuple like items, first field is JsonPath like path to
        identify the fields, second is the value, then fill in a table. Usually used to send a json or yaml content.

        NOTE: This method is not suite for too big data, we will have a upload function later.

        Args:
            category (str): Category name of current data.
            value (object): Object to save.
        """

        items = []

        # (path, item) tuples.
        stack = [("$", value)]

        # Do splitting and converting.
        while len(stack) > 0:
            cur_path, cur_item = stack.pop()

            cur_item_type = type(cur_item)

            if cur_item_type is dict:
                for k, v in cur_item.items():
                    stack.append((cur_path + f".{k}", v))
            elif cur_item_type is list \
                    or cur_item_type is tuple \
                    or (cur_item_type is torch.Tensor and cur_item.dim() > 1) \
                    or (cur_item_type is numpy.ndarray and len(cur_item.shape) > 1):

                for sub_index, sub_item in enumerate(cur_item):
                    stack.append((cur_path + f"[{sub_index}]", sub_item))
            elif cur_item_type is torch.Tensor:
                # We only accept 1 dim to json string.
                items.append({
                    "path": cur_path,
                    "value": json.dumps(cur_item.tolist())
                })
            else:
                items.append({
                    "path": cur_path,
                    "value": str(cur_item)
                })

        for item in items:
            self._put(MessageType.Data, (category, item))

    def close(self):
        """Close current client connection."""
        if self._is_started and self._sender is not None and self._sender.is_alive():
            # Send a close command and wait for stop.
            self._put(MessageType.Close, None)

            self._sender.join()

            self._is_started = False

    def _put(self, msg_type: MessageType, data: object):
        """Put data to queue to process in sender process.

        Args:
            msg_type (MessageType): Type of current message.
            data (object): Any data can be pickled.
        """
        self._data_queue.put((msg_type, data))

    def _start(self):
        """Start sender process, then we are ready to go."""
        self._sender = StreamitSender(self._data_queue, self._experiment_name, self._host)

        try:
            self._sender.start()

            self._is_started = True
        except Exception:
            print("Fail to start streamit client.")

    def __bool__(self):
        return True

    def __del__(self):
        self.close()

    def __enter__(self):
        """Support with statement."""
        self._start()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop after exit with statement."""
        self.close()


__all__ = ['StreamitClient']
