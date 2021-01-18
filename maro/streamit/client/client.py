

import json
import numpy
import torch
from .sender import StreamitSender

from multiprocessing import Queue, Process
from functools import partial
from typing import Union, List, Dict


from .common import MessageType


class Client:
    def __init__(self):
        self._sender: Process = None
        self._data_queue = Queue()

        self._cur_episode = 0
        self._cur_tick = 0

    def start(self, experiment_name: str, host="127.0.0.1"):
        self._experiment_name = experiment_name

        self._sender = StreamitSender(self._data_queue, self._experiment_name, host)

        self._sender.start()

    def info(self, scenario: str, topology: str, durations: int, total_episodes: int, **kwargs):
        self._put(MessageType.Experiment, (scenario, topology, durations, total_episodes, kwargs))

    def tick(self, tick: int):
        """Update current tick"""
        self._cur_tick = tick

        self._put(MessageType.Tick, tick)

    def episode(self, episode: int):
        """Update current episode"""
        self._cur_episode = episode

        self._put(MessageType.Episode, episode)

    def data(self, category: str, **kwargs):
        """Send data for sepcified category"""
        self._put(MessageType.Data, (category, kwargs))

    def complex(self, category: str, value: dict):
        """This method will split value dictionary into small items, that fill in a table.
        Usually used to send a json or yaml content.

        NOTE: This method is not suite for too big data, we will have a upload function later.

        Something like:

        item, value
        path.to.item, value
        path.to.item[0], value2
        """

        items = []

        stack = [("$", value)]  # (path, item) tuple

        while len(stack) > 0:
            cur_path, cur_item = stack.pop()

            cur_item_type = type(cur_item)

            if cur_item_type is dict:
                for k, v in cur_item.items():
                    stack.append((cur_path + f".{k}", v))
            elif cur_item_type is list  \
                    or cur_item_type is tuple   \
                    or (cur_item_type is torch.Tensor and cur_item.dim() > 1)   \
                    or (cur_item_type is numpy.ndarray and len(cur_item.shape) > 1):
                for sub_index, sub_item in enumerate(cur_item):
                    stack.append(
                        (cur_path + f"[{sub_index}]", sub_item)
                    )
            elif cur_item_type is torch.Tensor:
                # We only accept 1 dim to json string
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

    def upload(self, file: str, mode: str = None) -> str:
        """Upload a file to server and return a url path.

        Args:
            file (str): File path to upload.
            mode (str): Save mode:
                None: save under current experiment folder.
                "e": save to current episode folder
                "t": save to current tick folder

                Default is under experiment folder.
        """

        upload_path = self._experiment_name

        if mode == "e":
            upload_path = f"{self._experiment_name}/{self._cur_episode}"
        elif mode == "t":
            upload_path = f"{self._experiment_name}/{self._cur_episode}/{self._cur_tick}"

        self._put(MessageType.File, (file, mode))

        return upload_path

    def close(self):
        if self._sender is not None and self._sender.is_alive():
            print("waiting for sender stop")
            # send a close command and wait for stop
            self._put(MessageType.Close, None)

            self._sender.join()

    def _put(self, msg_type, data):
        self._data_queue.put((msg_type, data))

    def __getitem__(self, name: str):
        """Shorthand for category name, like: streamit["port_detail"](index=0, name="test")"""
        return partial(self.send, name)
