# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Any, Callable, Dict, Iterable, List, Optional

from maro.rl_v31.objects import ExpElement
from .typing import EnvStepRes, PolicyActionType
from .worker import EnvWorker
from .wrapper import EnvWrapper


class BaseVectorEnv(object):
    def __init__(
        self,
        env_fns: List[Callable[[], EnvWrapper]],  # TODO: use a single callable or a list of callables?
        worker_fn: Callable[[Callable[[], EnvWrapper]], EnvWorker],
        wait_num: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._env_fns = env_fns
        self.workers = [worker_fn(fn) for fn in env_fns]
        self.worker_cls = type(self.workers[0])

        self.env_num = len(env_fns)
        self.wait_num = wait_num or self.env_num
        assert 1 <= self.wait_num <= self.env_num

        self.timeout = timeout

        self.ready_ids = self.all_ids
        self.is_closed = False

    @property
    def all_ids(self) -> List[int]:
        return list(range(self.env_num))

    def _wrap_ids(self, ids: Optional[Iterable[int]]) -> List[int]:
        return self.all_ids if ids is None else list(ids)

    def reset(
        self,
        ids: Optional[Iterable[int]] = None,
        **kwargs: Any,
    ) -> Dict[int, EnvStepRes]:
        assert not self.is_closed

        ids = self._wrap_ids(ids)
        for i in ids:
            msg = {"func": "step", "kwargs": {**{"policy_act_dict": None}, **kwargs}}
            self.workers[i].send(msg)  # Send None to a worker means reset
        return {i: self.workers[i].recv()[1] for i in ids}

    def step(
        self, env_policy_acts: Dict[int, Dict[Any, PolicyActionType]], **kwargs: Any
    ) -> Dict[int, EnvStepRes]:
        assert not self.is_closed

        for i, policy_act in env_policy_acts.items():
            msg = {"func": "step", "kwargs": {**{"policy_act_dict": policy_act}, **kwargs}}
            self.workers[i].send(msg)

        return {i: self.workers[i].recv()[1] for i in env_policy_acts}  # TODO: process before return

    def gather_info(self, ids: Optional[Iterable[int]] = None) -> Dict[int, dict]:
        ids = self._wrap_ids(ids)
        for i in ids:
            msg = {"func": "gather_info", "kwargs": {}}
            self.workers[i].send(msg)
        return {i: self.workers[i].recv()[1] for i in ids}

    def collect_ready_exps(self, ids: Optional[Iterable[int]] = None) -> Dict[int, List[ExpElement]]:
        assert not self.is_closed

        ids = self._wrap_ids(ids)
        for i in ids:
            msg = {"func": "collect_ready_exps", "kwargs": {}}
            self.workers[i].send(msg)
        return {i: self.workers[i].recv()[1] for i in ids}
