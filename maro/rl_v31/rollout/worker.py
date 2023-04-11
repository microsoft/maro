# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from maro.rl_v31.objects import ExpElement

from .typing import EnvStepRes, PolicyActionType
from .wrapper import EnvWrapper


class EnvWorker(object, metaclass=ABCMeta):
    def __init__(self, env_func: Callable[[], EnvWrapper]) -> None:
        self._env_func = env_func

    @abstractmethod
    def send(self, msg: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def recv(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def step(self, policy_act_dict: Optional[Dict[Any, PolicyActionType]], **kwargs: Any) -> EnvStepRes:
        raise NotImplementedError

    @abstractmethod
    def collect_ready_exps(self) -> List[ExpElement]:
        raise NotImplementedError

    @abstractmethod
    def gather_info(self) -> dict:
        raise NotImplementedError


class DummyEnvWorker(EnvWorker):
    def __init__(self, env_func: Callable[[], EnvWrapper]) -> None:
        super().__init__(env_func=env_func)

        self._env = env_func()
        self._result: Any = None

    def send(self, msg: dict) -> None:
        func_name, kwargs = msg["func"], msg["kwargs"]
        func = getattr(self, func_name)
        self._result = func(**kwargs)

    def recv(self) -> Tuple[bool, Any]:
        return True, self._result

    def step(self, policy_act_dict: Optional[Dict[Any, PolicyActionType]], **kwargs: Any) -> EnvStepRes:
        return self._env.step(policy_act_dict)

    def collect_ready_exps(self) -> List[ExpElement]:
        return self._env.collect_ready_exps()

    def gather_info(self) -> dict:
        return self._env.gather_info()
