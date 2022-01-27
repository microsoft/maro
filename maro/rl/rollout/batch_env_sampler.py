# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import socket
import time
from itertools import chain
from typing import Dict, List, Tuple

import zmq
from zmq import Context, Poller

from maro.rl.utils.common import bytes_to_pyobj, pyobj_to_bytes
from maro.utils import DummyLogger, Logger


class BatchClient(object):
    """The communication client that the BatchEnvSampler uses to communicate with RolloutDispatcher.

    Args:
        name (str): Name of the client.
        address (Tuple[str, int]): Address (host and port) of the target dispatcher.
    """
    def __init__(self, name: str, address: Tuple[str, int]) -> None:
        self._name = name
        host, port = address
        self._dispatcher_ip = socket.gethostbyname(host)
        self._address = f"tcp://{self._dispatcher_ip}:{port}"
        self._poller = Poller()
        self._logger = Logger("batch_request_client")

    def collect(self, req: dict, parallelism: int, min_replies: int = None, grace_factor: float = None) -> List[dict]:
        """Send the requests to the dispatcher, collect the results from dispatcher, and send the results back.

        Args:
            req (dict): Request.
            parallelism (int): Number of workers we want to launch in parallel.
            min_replies (int, default=None): The current function will only return the results if we receives
                the results from at least min_replies workers. If it is None, we have to wait all workers return
                their results.
            grace_factor (float, default=None): Factor that determines the additional waiting time after we receives
                min_replies workers' results. For example, if we get min_replies workers' results within K minutes,
                we will wait another K * grace_factor minutes to see if we can collect more results.

        Returns:
            A list of results. Each element in the list is a dict that contains results from a worker.
        """
        if min_replies is None:
            min_replies = parallelism

        start_time = time.time()
        results = []
        req["parallelism"] = parallelism
        self._socket.send(pyobj_to_bytes(req))
        self._logger.info(f"{self._name} sent request")
        while len(results) < min_replies:
            result = self._socket.recv_multipart()
            results.append(bytes_to_pyobj(result[0]))

        if grace_factor is not None:
            countdown = int((time.time() - start_time) * grace_factor) * 1000  # milliseconds
            self._logger.info(f"allowing {countdown / 1000} seconds for remaining results")
            while len(results) < parallelism and countdown > 0:
                start = time.time()
                event = dict(self._poller.poll(countdown))
                if self._socket in event:
                    result = self._socket.recv_multipart()
                    result = bytes_to_pyobj(result[0])
                    assert isinstance(result, dict)
                    results.append(result)
                countdown -= time.time() - start

        self._logger.info(f"{self._name} received {min_replies} results")
        return results

    def close(self):
        """Close the client.
        """
        self._poller.unregister(self._socket)
        self._socket.disconnect(self._address)
        self._socket.close()

    def connect(self):
        """Establish the connection to dispatcher.
        """
        self._socket = Context.instance().socket(zmq.DEALER)
        self._socket.setsockopt_string(zmq.IDENTITY, self._name)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self._address)
        self._logger.info(f"connected to {self._address}")
        self._poller.register(self._socket, zmq.POLLIN)


class BatchEnvSampler:
    """BatchEnvSampler provides similar functionalities with AbsEnvSampler, but in parallelized fashion.
    Through BatchEnvSampler, we could run multiple env samplers (remotely) in parallel and get more experiences.

    Args:
        parallelism (int): Number of workers we want to launch in parallel.
        remote_address (Tuple[str, int]): Address (host and port) of the target dispatcher.
        min_env_samples (int, default=None): In each round of experience collection, we will wait until at least
            min_env_samples workers return their results. If it is None, we have to wait all workers return their
            results.
        grace_factor (float, default=None): Factor that determines the additional waiting time after we receives
            min_replies workers' results. For example, if we get min_replies workers' results within K minutes,
            we will wait another K * grace_factor minutes to see if we can collect more results.
        eval_parallelism (int, default=1): Number of workers we want to use in the evaluation.
        logger (Logger, default=DummyLogger): Logger.
    """
    def __init__(
        self,
        parallelism: int,
        remote_address: Tuple[str, int],
        min_env_samples: int = None,
        grace_factor: float = None,
        eval_parallelism: int = 1,
        logger: Logger = DummyLogger()
    ) -> None:
        if eval_parallelism > parallelism:
            raise ValueError(f"eval_parallelism cannot exceed the number of available workers: {parallelism}")

        super(BatchEnvSampler, self).__init__()
        self._client = BatchClient("batch_env_sampler", remote_address)
        self._logger = logger
        self._parallelism = parallelism
        self._min_env_samples = min_env_samples if min_env_samples is not None else self._parallelism
        self._grace_factor = grace_factor
        self._eval_parallelism = eval_parallelism

        self._ep = 0
        self._segment = 0
        self._end_of_episode = True

    def sample(self, policy_state: Dict[str, object] = None, num_steps: int = -1) -> dict:
        """Sample experiences.

        Args:
            policy_state (Dict[str, object]): Policy state dict. If it is not None, then we need to update all
                policies according to the latest policy states, then start the experience collection.
            num_steps (int, default=-1): Number of collecting steps. If it is -1, it means unlimited number of steps.

        Returns:
            A dict that contains the collected experiences and other additional information.
        """
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
        """Run testing and returns the testing results.

        Args:
            policy_state (Dict[str, object], default=None): Double-deck dict with format: {policy_name: policy_state}.
                If it is not None, it means we need to update all policies' state according to policy_state.

        Returns:
            A dict contains testing results.
        """
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
