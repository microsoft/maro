# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from typing import Dict, List, Tuple, Union

import numpy as np

from maro.simulator import DecisionMode

from .env_process import EnvProcess

ActionType = Union[Dict[int, object], List[object], object]


class VectorEnv:
    """Helper used to maintain several environment instances in different processes.

    NOTE:
        This helper do not care about if each environment has same tick (frame_index).
    """

    class SnapshotListNodeWrapper:
        """Wrapper to provide same interface as normal snapshot nodes.

        Args:
            env (VectorEnv): VectorEnv instance to send query command.
            node_name (str): Name of node bind to this wrapper, used for furthur querying.
        """
        def __init__(self, env, node_name: str):
            self.node_name = node_name
            self._env = env

        def __getitem__(self, args) -> List[np.ndarray]:
            return self._env._query(self.node_name, args)

    class SnapshotListWrapper:
        """Wrapper for snapshot list, used to provide same interface as normal snapshot list.

        Args:
            env (VectorEnv): VectorEnv instance used to send query command.
        """

        def __init__(self, env):
            self._env = env

        def __getitem__(self, node_name: str):
            return VectorEnv.SnapshotListNodeWrapper(self._env, node_name)

    def __init__(
        self,
        batch_num: int,
        scenario: str = None,
        topology: str = None,
        start_tick: int = 0,
        durations: int = 100,
        snapshot_resolution: int = 1,
        max_snapshots: int = None,
        decision_mode: DecisionMode = DecisionMode.Sequential,
        business_engine_cls: type = None,
        disable_finished_events: bool = False,
        options: dict = {}
    ):
        self._is_env_started = False

        # Ensure batch number less than CPU core
        assert 0 < batch_num <= os.cpu_count()

        self._env_pipes: List[Connection] = []
        self._pipes: List[Connection] = []
        self._sub_process_list: List[EnvProcess] = []

        self._batch_num = batch_num
        self._is_stopping = False
        self._snapshot_wrapper = VectorEnv.SnapshotListWrapper(self)

        self._start_environments(
            scenario,
            topology,
            start_tick,
            durations,
            snapshot_resolution,
            max_snapshots,
            decision_mode,
            business_engine_cls,
            disable_finished_events,
            options
        )

    @property
    def batch_number(self) -> int:
        """Int: Number of environment processes."""
        return self._batch_num

    @property
    def snapshot_list(self) -> SnapshotListWrapper:
        """SnapshotListWrapper: Snapshot list of environments, used to query states.
        The query result will be a list of numpy array."""
        return self._snapshot_wrapper

    @property
    def tick(self) -> List[int]:
        """List[int]: Return tick of all environments."""
        return self._send("tick")

    @property
    def frame_index(self) -> List[int]:
        """List[int]: Return frame_index of all environments."""
        return self._send("frame_index")

    def step(self, action: ActionType) -> Tuple[dict, object, bool]:
        """Push environments to next step.

        Args:
            action (ActionType): If action is a normal object, then it will be send to all environments as action.
            If it is a list, then its length must same as environment number, then will send to environments one by one.
            If it is a dict, then means we want to send action to specified environment,
            key is the index of environment, value is action.

        Returns:
            Tuple[dict, object, bool]: Tuple with: list of metrics, list of decision_events, is_done
        """
        if type(action) is list:
            assert len(action) == self._batch_num

        response_list = self._send("step", action)

        # Due with response
        metrics = []
        decision_events = []

        for resp in response_list:
            # Only is done when all
            metrics.append(resp[0])
            decision_events.append(resp[1])

        is_done = self._send("is_done")

        return metrics, decision_events, all(is_done)

    def reset(self):
        """Reset all the environments."""
        # Send reset command, and wait for response from all process.
        self._send("reset", wait_response=True)

    def stop(self):
        """Stop all environments."""
        if self._is_env_started and not self._is_stopping:
            self._is_stopping = True

            self._send("stop", wait_response=False)

            for proc in self._sub_process_list:
                proc.join()

            for pipe in (self._env_pipes + self._pipes):
                if not pipe.closed:
                    pipe.close()

    def __enter__(self):
        """Support with statement."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop after exit with statement."""
        self.stop()

    def __del__(self):
        """In case forget to call stop."""
        self.stop()

    def _query(self, node_name: str, args: slice):
        """Query state from each environments.

        Args:
            node_name (str): Node name to query.
            args (slice): Args for snapshot list querying.
        """
        return self._send("query", (node_name, args))

    def _send(self, cmd: str, content: Union[list, object, dict] = None, wait_response=True):
        """Send cmd and data to environments.

        Args:
            cmd (str): Command name to send.
            content (Union[list, object, dict]): Content to send.
            wait_response (bool): Wait for response from all environments?
        """
        content_type = type(content)

        # Pipes we sent message to.
        pipes = []

        for index, pipe in enumerate(self._pipes):
            if content_type is list:
                pipe.send((cmd, content[index]))

                pipes.append(pipe)
            elif content_type is dict:
                # Check if index exist.
                if index in content:
                    pipe.send((cmd, content[index]))

                    pipes.append(pipe)
            else:
                pipe.send((cmd, content))
                pipes.append(pipe)

        if wait_response:
            return [pipe.recv() for pipe in pipes]

        return None

    def _start_environments(self, *args, **kwargs):
        for i in range(self._batch_num):
            mp, sp = Pipe()

            self._pipes.append(mp)
            self._env_pipes.append(sp)

            env_proc = EnvProcess(sp, *args, **kwargs)

            self._sub_process_list.append(env_proc)

            env_proc.start()

        self._is_env_started = True
