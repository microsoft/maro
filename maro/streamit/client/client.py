
from .sender import StreamitSender

from multiprocessing import Queue, Process
from functools import partial
from typing import Union, List, Dict


from .common import MessageType


class Client:
    def __init__(self, experiment_name: str):
        self._experiment_name = experiment_name
        self._sender: Process = None
        self._data_queue = Queue()

        self._cur_episode = 0
        self._cur_tick = 0

    def start(self, host="127.0.0.1"):
        self._sender = StreamitSender(
            self._data_queue, self._experiment_name, host)

        self._sender.start()

    def info(self, scenario: str, topology: str, durations: int, total_episodes: int, **kwargs):
        self._put(MessageType.Experiment, (scenario, topology,
                                           durations, total_episodes, kwargs))

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

    def dict(self, category: str, value: dict):
        """This method will split value dictionary into small items, that fill in a table.
        Usually used to send a json or yaml content.

        NOTE: This method is not suite for too big data, we will have a upload function later.

        Something like:

        item, value
        path.to.item, value
        path.to.item[0], value2
        """

        items = []
        flat_dict(value, items)

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


# TODO: reduce recurcive calling with stack
def flat_dict(d: dict, result_list: list, path: str = None):
    for k, v in d.items():
        sub_path = path
        if sub_path is None:
            sub_path = k
        else:
            sub_path += f".{k}"

        v_type = type(v)

        if v_type is dict:
            flat_dict(v, result_list, sub_path)
        elif v_type is list or v_type is tuple:
            flat_list(v, result_list, sub_path)
        else:
            result_list.append({"path": sub_path, "value": str(v)})
        # NOTE: we assuming that dict only contains raw data type, list, tuple or dict, no customized time.


def flat_list(l: list, result_list: list, path: str = None):
    for i, item in enumerate(l):
        sub_path = path

        if sub_path is None:
            sub_path = ""

        item_type = type(item)

        if item_type is dict:
            flat_dict(item, result_list, sub_path + f"[{i}]")
        elif item_type is list or item_type is tuple:
            flat_list(item, result_list, sub_path + f"[{i}]")
        else:
            result_list.append(
                {"path": sub_path + "[]", "index": i, "value": str(item)})
