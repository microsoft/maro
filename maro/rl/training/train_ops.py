# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import inspect
import socket
from abc import ABCMeta, abstractmethod
from typing import Callable, Tuple

import torch
import zmq
from zmq.asyncio import Context, Poller

from maro.rl.policy import RLPolicy
from maro.rl.utils import AbsTransitionBatch, MultiTransitionBatch, TransitionBatch
from maro.rl.utils.common import bytes_to_pyobj, pyobj_to_bytes
from maro.utils import DummyLogger, Logger


class AbsTrainOps(object, metaclass=ABCMeta):
    """The basic component for training a policy, which takes charge of loss / gradient computation and policy update.
    Each ops is used for training a single policy. An ops is an atomic unit in the distributed mode.

    Args:
        device (str): Identifier for the torch device. The policy will be moved to the specified device.
            If it is None, the device will be set to "cpu" if cuda is unavailable and "cuda" otherwise.
        is_single_scenario (bool): Identifier of whether this ops is used under a single trainer or a multi trainer.
        get_policy_func (Callable[[], RLPolicy]): Function used to create the policy of this ops.
    """

    def __init__(
        self,
        name: str,
        device: str,
        is_single_scenario: bool,
        get_policy_func: Callable[[], RLPolicy]
    ) -> None:
        super(AbsTrainOps, self).__init__()
        self._name = name
        self._device = torch.device(device) if device is not None \
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_single_scenario = is_single_scenario

        # Create the policy and put it on the right device.
        if self._is_single_scenario:
            self._policy = get_policy_func()
            self._policy.to_device(self._device)

    @property
    def name(self) -> str:
        return self._name

    @property
    def policy_state_dim(self) -> int:
        return self._policy.state_dim

    @property
    def policy_action_dim(self) -> int:
        return self._policy.action_dim

    def _is_valid_transition_batch(self, batch: AbsTransitionBatch) -> bool:
        """Used to check the transition batch's type. If this ops is used under a single trainer, the batch should be
        a `TransitionBatch`. Otherwise, it should be a `MultiTransitionBatch`.

        Args:
            batch (AbsTransitionBatch): The batch to be checked.
        """
        return isinstance(batch, TransitionBatch) if self._is_single_scenario \
            else isinstance(batch, MultiTransitionBatch)

    @abstractmethod
    def get_state(self) -> dict:
        """Get the train ops's state.

        Returns:
            A dict that contains ops's state.
        """
        raise NotImplementedError

    @abstractmethod
    def set_state(self, ops_state_dict: dict) -> None:
        """Set ops's state.

        Args:
            ops_state_dict (dict): New ops state.
        """
        raise NotImplementedError

    def get_policy_state(self) -> Tuple[str, object]:
        """Get the policy's state.

        Returns:
            policy_name (str)
            policy_state (object)
        """
        return self._policy.name, self._policy.get_state()

    def set_policy_state(self, policy_state: object) -> None:
        """Update the policy's state.

        Args:
            policy_state (object): The policy state.
        """
        self._policy.set_state(policy_state)


def remote(func) -> Callable:
    """Annotation to indicate that an function / method can be called remotely
    """
    def remote_annotate(*args, **kwargs) -> object:
        return func(*args, **kwargs)

    return remote_annotate


class AsyncClient(object):
    """The async communication client that the trainer uses to communicate with TrainOpsDispatcher.

    Args:
        name (str): Name of the client.
        address (Tuple[str, int]): Address (host and port) of the target dispatcher.
        logger (Logger, default=None): logger.
    """
    def __init__(self, name: str, address: Tuple[str, int], logger: Logger = None) -> None:
        self._name = name
        host, port = address
        self._dispatcher_ip = socket.gethostbyname(host)
        self._address = f"tcp://{self._dispatcher_ip}:{port}"
        self._logger = DummyLogger() if logger is None else logger

    async def send_request(self, req: dict) -> None:
        """Send the request to dispatcher.

        Args:
            req (dict): Request.
        """
        await self._socket.send(pyobj_to_bytes(req))
        self._logger.debug(f"{self._name} sent request {req['func']}")

    async def get_response(self) -> object:
        """Listening the socket and return the received result.
        """
        while True:
            events = await self._poller.poll(timeout=100)
            if self._socket in dict(events):
                result = await self._socket.recv_multipart()
                self._logger.debug(f"{self._name} received result")
                return bytes_to_pyobj(result[0])

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
        self._logger.debug(f"connected to {self._address}")
        self._poller = Poller()
        self._poller.register(self._socket, zmq.POLLIN)


class RemoteOps(object):
    """Ops for remote executing. RemoteOps provides similar interfaces to TrainOps, but instead of doing all
    calculations locally, RemoteOps will only execute "local" methods locally. For "remote" methods, RemoteOps
    will send a request to dispatcher and let the remote workers to do the actual calculation.

    Args:
        ops (AbsTrainOps): The train ops. This train ops instance is used to store ops state locally and execute
            all "local" (not "remote") methods.
        address (Tuple[str, int]): Address (host and port) of the target dispatcher.
        logger (Logger, default=None): logger.
    """
    def __init__(self, ops: AbsTrainOps, address: Tuple[str, int], logger: Logger = None) -> None:
        self._ops = ops
        self._client = AsyncClient(self._ops.name, address, logger=logger)
        self._client.connect()

    def __getattribute__(self, attr_name: str) -> object:
        # Ignore methods that belong to the parent class
        try:
            return super().__getattribute__(attr_name)
        except AttributeError:
            pass

        def remote_method(ops_state, func_name: str, client: AsyncClient) -> Callable:
            async def remote_call(*args, **kwargs) -> object:
                req = {"state": ops_state, "func": func_name, "args": args, "kwargs": kwargs}
                await client.send_request(req)
                response = await client.get_response()
                return response

            return remote_call

        attr = getattr(self._ops, attr_name)
        if inspect.ismethod(attr) and attr.__name__ == "remote_annotate":
            return remote_method(self._ops.get_state(), attr_name, self._client)

        return attr

    def exit(self) -> None:
        """Close the remote ops.
        """
        self._client.close()
