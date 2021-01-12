
import os
import struct
import json
import contextlib

from typing import Union
from multiprocessing import Process, Queue

from .common import MessageType, DataType
from .sender import ExperimentDataSender


class ExperimentDataStream:
    def __init__(self, experiment_name: str):
        assert experiment_name is not None

        self._sender: ExperimentDataSender = None
        self._data_queue = Queue()

        # Used to mark is current data is time depended
        self._is_time_depend_data = True

        # category name -> is time depend
        self._category_info = {}

        # episode -> tick -> data list
        self._cache = []

        self._experiment_name = experiment_name
        self._cur_episode = 0
        self._cur_tick = 0

    def experiment_info(self, scenario: str, topology: str, total_episodes: int, durations: int, start_tick: int = 0):
        self._send(
            MessageType.BeginExperiment,
            (scenario, topology, total_episodes, start_tick, durations)
        )

    def episode(self, episode: int):
        self._cur_episode = episode

        self._send(MessageType.BeginEpisode, episode)

    def tick(self, tick: int):
        # send first, then update current tick
        if len(self._cache) > 0:
            cached_data = self._cache
            self._cache = []
            self._send(
                MessageType.Data,
                (self._cur_episode, self._cur_tick, cached_data)
            )

        self._cur_tick = tick

    def category(self, name: str, fields: dict):
        self._send(MessageType.Category, (name, fields))

    def row(self, category, *args):
        self._cache.append((category, *args))

    def start(self, address="127.0.0.1", port="8812", user="admin", password="quest", database="qdb"):
        """Connect to server, prepare to send data"""
        self._sender = ExperimentDataSender(
            self._data_queue, self._experiment_name, address, port, user, password, database
        )

        self._sender.start()

    def close(self):
        self._data_queue.put_nowait("stop")

        self._sender.join()

    def _send(self, type: MessageType, data):
        try:
            self._data_queue.put_nowait((type, data))
        except Exception as ex:
            # If reach the limitation, then it will exception
            print(ex)

    def __del__(self):
        print("gc collecting")
        if self._sender is not None and self._sender.is_alive:
            print("waiting for close")
            self._sender.terminate()
