# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import copy
import os
import shutil
import typing
from collections import defaultdict
from typing import List, Optional

import pandas as pd

from maro.rl_v31.rollout.wrapper import AgentWrapper
from maro.rl_v31.training.training_manager import TrainingManager
from maro.utils import LoggerV2

if typing.TYPE_CHECKING:
    from maro.rl_v31.workflow.workflow import Workflow


class Callback(object):
    def __init__(self) -> None:
        self.workflow: Optional[Workflow] = None
        self.agent_wrapper: Optional[AgentWrapper] = None
        self.training_manager: Optional[TrainingManager] = None
        self.logger: Optional[LoggerV2] = None

    def on_episode_start(self, ep: int) -> None:
        pass

    def on_episode_end(self, ep: int) -> None:
        pass

    def on_training_start(self, ep: int) -> None:
        pass

    def on_training_end(self, ep: int) -> None:
        pass

    def on_validation_start(self, ep: int) -> None:
        pass

    def on_validation_end(self, ep: int) -> None:
        pass

    def on_test_start(self, ep: int) -> None:
        pass

    def on_test_end(self, ep: int) -> None:
        pass


class EarlyStopping(Callback):
    def __init__(self, patience: int, monitor: str, higher_better: bool) -> None:
        super().__init__()

        self._patience = patience
        self._monitor = monitor
        self._higher_better = higher_better

        self.best_ep: int = -1
        self.best: float = float("-inf") if higher_better else float("inf")

    def on_validation_end(self, ep: int) -> None:
        cur = self.workflow.valid_metrics[self._monitor]
        if (self._higher_better and cur > self.best) or (not self._higher_better and cur < self.best):
            self.best_ep = ep
            self.best = cur
        self.logger.info(f"Current metric: {cur} @ ep {ep}. Best metric: {self.best} @ ep {self.best_ep}")

        if ep - self.best_ep > self._patience:
            self.workflow.early_stop = True
            self.logger.info(
                f"Validation metric has not been updated for {ep - self.best_ep} "
                f"epochs (patience = {self._patience} epochs). Early stop.",
            )


class Checkpoint(Callback):
    def __init__(self, path: str, interval: int) -> None:
        super().__init__()

        self._path = path
        self._interval = interval

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    def on_training_end(self, ep: int) -> None:
        if ep % self._interval == 0:
            ep_path = os.path.join(self._path, str(ep))
            os.makedirs(ep_path, exist_ok=True)

            self.agent_wrapper.save(ep_path)
            self.training_manager.save(ep_path)
            self.logger.info(f"[Episode {ep}] All policy/trainer states saved under {self._path}")

            self.make_copy(str(ep), "latest")

    def make_copy(self, src: str, dst: str) -> None:
        src_path = os.path.join(self._path, src)
        dst_path = os.path.join(self._path, dst)
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        shutil.copytree(src_path, dst_path)


class MetricsRecorder(Callback):
    def __init__(self, path: str) -> None:
        super().__init__()

        self._path = path
        self._first = defaultdict(lambda: True)

    def _dump(self, ep: int, metrics: dict, tag: str) -> None:
        metrics["ep"] = ep
        first = self._first[tag]
        path = os.path.join(self._path, f"metrics_{tag}.csv")

        df = pd.DataFrame.from_records([metrics]).set_index(["ep"])
        df.to_csv(path, index=True, header=first, mode="w" if first else "a")

        self._first[tag] = False

    def on_training_end(self, ep: int) -> None:
        train_metrics = copy.deepcopy(self.workflow.train_metrics)
        self._dump(ep, train_metrics, "full")

    def on_validation_end(self, ep: int) -> None:
        train_metrics = copy.deepcopy(self.workflow.train_metrics)
        valid_metrics = copy.deepcopy(self.workflow.valid_metrics)
        valid_metrics = {"val/" + str(k): v for k, v in valid_metrics.items()}
        self._dump(ep, {**train_metrics, **valid_metrics}, "valid")


class CallbackManager(object):
    def __init__(
        self,
        workflow: Workflow,
        callbacks: List[Callback],
        agent_wrapper: AgentWrapper,
        training_manager: TrainingManager,
        logger: LoggerV2,
    ) -> None:
        super().__init__()

        self._callbacks = callbacks
        for callback in self._callbacks:
            callback.workflow = workflow
            callback.agent_wrapper = agent_wrapper
            callback.training_manager = training_manager
            callback.logger = logger

    def on_episode_start(self, ep: int) -> None:
        for callback in self._callbacks:
            callback.on_episode_start(ep)

    def on_episode_end(self, ep: int) -> None:
        for callback in self._callbacks:
            callback.on_episode_end(ep)

    def on_training_start(self, ep: int) -> None:
        for callback in self._callbacks:
            callback.on_training_start(ep)

    def on_training_end(self, ep: int) -> None:
        for callback in self._callbacks:
            callback.on_training_end(ep)

    def on_validation_start(self, ep: int) -> None:
        for callback in self._callbacks:
            callback.on_validation_start(ep)

    def on_validation_end(self, ep: int) -> None:
        for callback in self._callbacks:
            callback.on_validation_end(ep)

    def on_test_start(self, ep: int) -> None:
        for callback in self._callbacks:
            callback.on_test_start(ep)

    def on_test_end(self, ep: int) -> None:
        for callback in self._callbacks:
            callback.on_test_end(ep)
