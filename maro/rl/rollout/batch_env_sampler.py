# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from itertools import chain
from typing import Dict, List, Tuple

import zmq
from zmq import Context, Poller

from maro.rl.utils.common import bytes_to_pyobj, get_ip_address_by_hostname, pyobj_to_bytes
from maro.utils import DummyLogger, Logger


class BatchClient(object):
    def __init__(self, name: str, address: Tuple[str, int], logger: Logger = None) -> None:
        self._name = name
        host, port = address
        self._proxy_ip = get_ip_address_by_hostname(host)
        self._address = f"tcp://{self._proxy_ip}:{port}"
        self._poller = Poller()
        self._logger = logger

    def collect(self, req: dict, parallelism: int, min_replies: int = None, grace_factor: int = None) -> List[dict]:
        if min_replies is None:
            min_replies = parallelism

        start_time = time.time()
        results = []
        req["parallelism"] = parallelism
        self._socket.send(pyobj_to_bytes(req))
        self._logger.debug(f"{self._name} sent request")
        while len(results) < min_replies:
            result = self._socket.recv_multipart()
            results.append(bytes_to_pyobj(result[0]))

        if grace_factor is not None:
            countdown = int((time.time() - start_time) * grace_factor) * 1000  # milliseconds
            self._logger.debug(f"allowing {countdown / 1000} seconds for remaining results")
            while len(results) < parallelism and countdown > 0:
                start = time.time()
                event = dict(self._poller.poll(countdown))
                if self._socket in event:
                    result = self._socket.recv_multipart()
                    result = bytes_to_pyobj(result[0])
                    assert isinstance(result, dict)
                    results.append(result)
                countdown -= time.time() - start

        self._logger.debug(f"{self._name} received {len(results)} results")
        return results

    def close(self):
        self._poller.unregister(self._socket)
        self._socket.disconnect(self._address)
        self._socket.close()

    def connect(self):
        self._context = Context.instance()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt_string(zmq.IDENTITY, self._name)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self._address)
        self._logger.info(f"connected to {self._address}")
        self._poller.register(self._socket, zmq.POLLIN)

    def exit(self):
        self._socket.send(b"EXIT")


class BatchEnvSampler:
    def __init__(
        self,
        parallelism: int,
        proxy_address: Tuple[str, int],
        min_env_samples: int = None,
        grace_factor: float = None,
        eval_parallelism: int = 1,
        logger: Logger = None
    ) -> None:
        if eval_parallelism > parallelism:
            raise ValueError(f"eval_parallelism cannot exceed the number of available workers: {parallelism}")

        super(BatchEnvSampler, self).__init__()
        self._logger = logger if logger else DummyLogger()
        self._client = BatchClient("batch_env_sampler", proxy_address, logger=logger)

        self._parallelism = parallelism
        self._min_env_samples = min_env_samples if min_env_samples is not None else self._parallelism
        self._grace_factor = grace_factor
        self._eval_parallelism = eval_parallelism

        self._ep = 0
        self._segment = 0
        self._end_of_episode = True

    def sample(
        self, policy_state: Dict[str, object] = None, num_steps: int = -1
    ) -> dict:
        if self._end_of_episode:
            self._ep += 1
            self._segment = 1
        else:
            self._segment += 1
        self._logger.info(f"Collecting roll-out data for episode {self._ep}, segment {self._segment}")
        self._client.connect()
        req = {
            "type": "sample", "policy_state": policy_state, "num_steps": num_steps, "parallelism": self._parallelism
        }
        results = self._client.collect(
            req, self._parallelism,
            min_replies=self._min_env_samples,
            grace_factor=self._grace_factor
        )
        self._client.close()
        self._end_of_episode = any(res["end_of_episode"] for res in results)
        merged_experiences = list(chain(*[res["experiences"] for res in results]))  # List[List[ExpElement]]
        return {
            "end_of_episode": self._end_of_episode,
            "experiences": merged_experiences,
            "info": [res["info"][0] for res in results]
        }

    def test(self, policy_state: Dict[str, object] = None) -> dict:
        self._client.connect()
        req = {
            "type": "test",
            "policy_state": policy_state,
            "parallelism": self._eval_parallelism
        }
        results = self._client.collect(req, self._eval_parallelism)
        self._client.close()
        return {
            "info": [res["info"][0] for res in results]
        }

    def exit(self):
        self._client.connect()
        self._client.exit()
