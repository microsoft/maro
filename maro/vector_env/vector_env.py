# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pickle
import numpy as np

from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from typing import List, Tuple, Union

from maro.simulator import DecisionMode

from .mpenv_wrapper import MPEnvWrapper


class SnapshotQueryNodeWrapper:
    def __init__(self,  env, node_name: str):
        self.node_name = node_name
        self._env = env

    def __getitem__(self, args) -> List[np.ndarray]:
        return self._env._query(self.node_name, args)

class SnapshotQueryWrapper:
    def __init__(self, env):
        self._env = env

    def __getitem__(self, node_name: str) -> SnapshotQueryNodeWrapper:
        return SnapshotQueryNodeWrapper(self._env, node_name)


class VectorEnv:
    def __init__(self, batch_num: int, sync_tick=False,
                 scenario: str = None, topology: str = None,
                 start_tick: int = 0, durations: int = 100, snapshot_resolution: int = 1, max_snapshots: int = None,
                 decision_mode: DecisionMode = DecisionMode.Sequential,
                 business_engine_cls: type = None, disable_finished_events: bool = False,
                 options: dict = {}):
        # Ensure batch number less than CPU core
        assert batch_num <= os.cpu_count()

        self._env_pipes: List[Connection] = []
        self._pipes: List[Connection] = []
        self._sub_process_list: List[MPEnvWrapper] = []

        self._batch_num = batch_num
        self._is_stopping = False
        self._snapshot_wrapper = SnapshotQueryWrapper(self)

        self._start_environments(
            scenario, topology, start_tick, durations, snapshot_resolution, max_snapshots,
            decision_mode, business_engine_cls, disable_finished_events, options
        )

    @property
    def snapshot_list(self):
        return self._snapshot_wrapper

    def step(self, action: Union[List[object], object]) -> List[Tuple[int, dict, object]]:
        # call step on each environemnts
        if type(action) is list:
            assert len(action) == len(self._batch_num)

        response_list = self._send("step", action)

        # Due with response
        # Combine is_done
        is_done = True
        metrics = []
        decision_events = []

        for resp in response_list:
            # Only is done when all
            is_done = is_done and resp[2]

            metrics.append(resp[0])
            decision_events.append(resp[1])

        return metrics, decision_events, is_done

    def reset(self):
        # send reset command, and wait for response from all process
        self._send("reset", wait_response=True)

    def stop(self):
        if not self._is_stopping:
            self._is_stopping = True

            self._send("stop", wait_response=False)

            for proc in self._sub_process_list:
                proc.join()

            for pipe in (self._env_pipes + self._pipes):
                if not pipe.closed:
                    pipe.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def __del__(self):
        # In case forget to call stop
        self.stop()

    def _query(self, node_name: str, args):
        return self._send("query", (node_name, args))

    def _send(self, cmd: str, content: Union[list, str] = None, wait_response=True):
        content_type = type(content)

        for index, pipe in enumerate(self._pipes):
            if content_type is list:
                pipe.send((cmd, content[list]))
            else:
                pipe.send((cmd, content))

        if wait_response:
            return [pipe.recv() for pipe in self._pipes]

        return None

    def _start_environments(self, scenario: str = None, topology: str = None,
                            start_tick: int = 0, durations: int = 100, snapshot_resolution: int = 1, max_snapshots: int = None,
                            decision_mode: DecisionMode = DecisionMode.Sequential,
                            business_engine_cls: type = None, disable_finished_events: bool = False,
                            options: dict = {}):
        for i in range(self._batch_num):
            mp, sp = Pipe()

            self._pipes.append(mp)
            self._env_pipes.append(sp)

            env_proc = MPEnvWrapper(
                sp, scenario, topology, start_tick,
                durations, snapshot_resolution, max_snapshots,
                decision_mode, business_engine_cls,
                disable_finished_events, options
            )

            self._sub_process_list.append(env_proc)

            env_proc.start()
