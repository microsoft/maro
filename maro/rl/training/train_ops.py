# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import inspect
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Optional, Tuple, Union

import torch
import zmq
from zmq.asyncio import Context, Poller

from maro.rl.policy import RLPolicy
from maro.rl.utils.common import bytes_to_pyobj, get_ip_address_by_hostname, pyobj_to_bytes
from maro.utils import DummyLogger, LoggerV2


class AbsTrainOps(object, metaclass=ABCMeta):
    """The basic component for training a policy, which takes charge of loss / gradient computation and policy update.
    Each ops is used for training a single policy. An ops is an atomic unit in the distributed mode.

    Args:
        name (str): Name of the ops. This is usually a policy name.
        policy (RLPolicy): Policy instance.
        parallelism (int, default=1): Desired degree of data parallelism.
    """

    def __init__(
        self,
        name: str,
        policy: RLPolicy,
        parallelism: int = 1,
    ) -> None:
        super(AbsTrainOps, self).__init__()
        self._name = name
        self._policy = policy
        self._parallelism = parallelism
        self._device: Optional[torch.device] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def policy_state_dim(self) -> int:
        return self._policy.state_dim

    @property
    def policy_action_dim(self) -> int:
        return self._policy.action_dim

    @property
    def parallelism(self) -> int:
        return self._parallelism

    def get_state(self) -> dict:
        """Get the train ops's state.

        Returns:
            A dict that contains ops's state.
        """
        return {
            "policy": self.get_policy_state(),
            "non_policy": self.get_non_policy_state(),
        }

    def set_state(self, ops_state_dict: dict) -> None:
        """Set ops's state.

        Args:
            ops_state_dict (dict): New ops state.
        """
        assert ops_state_dict["policy"][0] == self._policy.name
        self.set_policy_state(ops_state_dict["policy"][1])
        self.set_non_policy_state(ops_state_dict["non_policy"])

    def get_policy_state(self) -> Tuple[str, dict]:
        """Get the policy's state.

        Returns:
            policy_name (str)
            policy_state (Any)
        """
        return self._policy.name, self._policy.get_state()

    def set_policy_state(self, policy_state: dict) -> None:
        """Update the policy's state.

        Args:
            policy_state (dict): The policy state.
        """
        self._policy.set_state(policy_state)

    @abstractmethod
    def get_non_policy_state(self) -> dict:
        """Get states other than policy.

        Returns:
            A dict that contains non-policy state.
        """
        raise NotImplementedError

    @abstractmethod
    def set_non_policy_state(self, state: dict) -> None:
        """Set states other than policy.

        Args:
            state (dict): Non-policy state.
        """
        raise NotImplementedError

    @abstractmethod
    def to_device(self, device: str = None) -> None:
        raise NotImplementedError


def remote(func: Callable) -> Callable:
    """Annotation to indicate that a function / method can be called remotely.

    This annotation takes effect only when an ``AbsTrainOps`` object is wrapped by a ``RemoteOps``.
    """

    def remote_annotate(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return remote_annotate


class AsyncClient(object):
    """Facility used by a ``RemoteOps`` instance to communicate asynchronously with ``TrainingProxy``.

    Args:
        name (str): Name of the client.
        address (Tuple[str, int]): Address (host and port) of the training proxy.
        logger (LoggerV2, default=None): logger.
    """

    def __init__(self, name: str, address: Tuple[str, int], logger: LoggerV2 = None) -> None:
        self._logger: Union[LoggerV2, DummyLogger] = logger if logger is not None else DummyLogger()
        self._name = name
        host, port = address
        self._proxy_ip = get_ip_address_by_hostname(host)
        self._address = f"tcp://{self._proxy_ip}:{port}"
        self._logger.info(f"Proxy address: {self._address}")

    async def send_request(self, req: dict) -> None:
        """Send a request to the proxy in asynchronous fashion.

        This is a coroutine and is executed asynchronously with calls to other AsyncClients' ``send_request`` calls.

        Args:
            req (dict): Request that contains task specifications and parameters.
        """
        await self._socket.send(pyobj_to_bytes(req))
        self._logger.debug(f"{self._name} sent request {req['func']}")

    async def get_response(self) -> Any:
        """Waits for a result in asynchronous fashion.

        This is a coroutine and is executed asynchronously with calls to other AsyncClients' ``get_response`` calls.
        This ensures that all clients' tasks are sent out as soon as possible before the waiting for results starts.
        """
        while True:
            events = await self._poller.poll(timeout=100)
            if self._socket in dict(events):
                result = await self._socket.recv_multipart()
                self._logger.debug(f"{self._name} received result")
                return bytes_to_pyobj(result[0])

    def close(self) -> None:
        """Close the connection to the proxy."""
        self._poller.unregister(self._socket)
        self._socket.disconnect(self._address)
        self._socket.close()

    def connect(self) -> None:
        """Establish the connection to the proxy."""
        self._socket = Context.instance().socket(zmq.DEALER)
        self._socket.setsockopt_string(zmq.IDENTITY, self._name)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self._address)
        self._logger.debug(f"connected to {self._address}")
        self._poller = Poller()
        self._poller.register(self._socket, zmq.POLLIN)

    async def exit(self) -> None:
        """Send EXIT signals to the proxy indicating no more tasks."""
        await self._socket.send(b"EXIT")


class RemoteOps(object):
    """Wrapper for ``AbsTrainOps``.

    RemoteOps provides similar interfaces to ``AbsTrainOps``. Any method annotated by the remote decorator in the
    definition of the train ops is transformed to a remote method. Calling this method invokes using the internal
    ``AsyncClient`` to send the required task parameters to a ``TrainingProxy`` that handles task dispatching and
    result collection. Methods not annotated by the decorator are not affected.

    Args:
        ops (AbsTrainOps): An ``AbsTrainOps`` instance to be wrapped. Any method annotated by the remote decorator in
            its definition is transformed to a remote function call.
        address (Tuple[str, int]): Address (host and port) of the training proxy.
        logger (LoggerV2, default=None): logger.
    """

    def __init__(self, ops: AbsTrainOps, address: Tuple[str, int], logger: LoggerV2 = None) -> None:
        self._ops = ops
        self._client = AsyncClient(self._ops.name, address, logger=logger)
        self._client.connect()

    def __getattribute__(self, attr_name: str) -> Any:
        # Ignore methods that belong to the parent class
        try:
            return super().__getattribute__(attr_name)
        except AttributeError:
            pass

        def remote_method(ops_state: Any, func_name: str, desired_parallelism: int, client: AsyncClient) -> Callable:
            async def remote_call(*args: Any, **kwargs: Any) -> Any:
                req = {
                    "state": ops_state,
                    "func": func_name,
                    "args": args,
                    "kwargs": kwargs,
                    "desired_parallelism": desired_parallelism,
                }
                await client.send_request(req)
                response = await client.get_response()
                return response

            return remote_call

        attr = getattr(self._ops, attr_name)
        if inspect.ismethod(attr) and attr.__name__ == "remote_annotate":
            return remote_method(self._ops.get_state(), attr_name, self._ops.parallelism, self._client)

        return attr

    async def exit(self) -> None:
        """Close the internal task client."""
        await self._client.exit()
