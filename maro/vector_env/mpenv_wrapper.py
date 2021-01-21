# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle
from multiprocessing import Process
from multiprocessing.connection import Connection

from maro.simulator import Env, DecisionMode


class MPEnvWrapper(Process):
    def __init__(self, pipe: Connection,
                 *args, **kwargs):
        super().__init__()
        self._pipe = pipe

        self._env: Env = None
        self._args = args
        self._kwargs = kwargs

    def run(self):
        metrics = None
        decision_event = None,
        is_done = False

        env =  Env(*self._args, **self._kwargs)

        while True:
            cmd, content = self._pipe.recv()

            if cmd == "step":
                metrics, decision_event, is_done = env.step(content)

                self._pipe.send((metrics, decision_event, is_done, env.frame_index))
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
            elif cmd == "stop":
                self._pipe.send(None)
                break
