# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from multiprocessing import Process
from multiprocessing.connection import Connection

from maro.simulator import Env


class EnvProcess(Process):
    """Wrapper for envrioment process,

    Args:
        pipe (Connection): Pipe that used to communicate between main process and this process.
        (args, kwargs): Parameter for Env class.
    """

    def __init__(self, pipe: Connection, *args, **kwargs):
        super().__init__()

        self._pipe = pipe
        self._env: Env = None
        self._args = args
        self._kwargs = kwargs

    def run(self):
        """Initialize environment and process commands."""
        metrics = None
        decision_event = None,
        is_done = False

        env = Env(*self._args, **self._kwargs)

        while True:
            cmd, content = self._pipe.recv()

            if cmd == "step":
                if is_done:
                    # Skip is current environment is completed.
                    self._pipe.send((None, None, True, env.frame_index))
                else:
                    metrics, decision_event, is_done = env.step(content)

                    self._pipe.send((metrics, decision_event))
            elif cmd == "reset":
                env.reset()

                metrics = None
                decision_event = None
                is_done = False

                self._pipe.send(None)
            elif cmd == "query":
                node_name, args = content

                states = env.snapshot_list[node_name][args]

                self._pipe.send(states)
            elif cmd == "tick":
                self._pipe.send(env.tick)
            elif cmd == "frame_index":
                self._pipe.send(env.frame_index)
            elif cmd == "is_done":
                self._pipe.send(is_done)
            elif cmd == "stop":
                self._pipe.send(None)
                break
