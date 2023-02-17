# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import copy
import os
import typing
from typing import Dict, List, Optional, Union

import pandas as pd

from maro.rl.rollout import AbsEnvSampler, BatchEnvSampler
from maro.rl.training import TrainingManager
from maro.utils import LoggerV2

if typing.TYPE_CHECKING:
    from maro.rl.workflows.main import TrainingWorkflow

EnvSampler = Union[AbsEnvSampler, BatchEnvSampler]


class Callback(object):
    def __init__(self) -> None:
        self.workflow: Optional[TrainingWorkflow] = None
        self.env_sampler: Optional[EnvSampler] = None
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
    def __init__(self, patience: int) -> None:
        super(EarlyStopping, self).__init__()

        self._patience = patience
        self._best_ep: int = -1
        self._best: float = float("-inf")

    def on_validation_end(self, ep: int) -> None:
        cur = self.env_sampler.monitor_metrics()
        if cur > self._best:
            self._best_ep = ep
            self._best = cur
        self.logger.info(f"Current metric: {cur} @ ep {ep}. Best metric: {self._best} @ ep {self._best_ep}")

        if ep - self._best_ep > self._patience:
            self.workflow.early_stop = True
            self.logger.info(
                f"Validation metric has not been updated for {ep - self._best_ep} "
                f"epochs (patience = {self._patience} epochs). Early stop.",
            )


class Checkpoint(Callback):
    def __init__(self, path: str, interval: int) -> None:
        super(Checkpoint, self).__init__()

        self._path = path
        self._interval = interval

    def on_training_end(self, ep: int) -> None:
        if ep % self._interval == 0:
            self.training_manager.save(os.path.join(self._path, str(ep)))
            self.logger.info(f"[Episode {ep}] All trainer states saved under {self._path}")


class MetricsRecorder(Callback):
    def __init__(self, path: str) -> None:
        super(MetricsRecorder, self).__init__()

        self._full_metrics: Dict[int, dict] = {}
        self._valid_metrics: Dict[int, dict] = {}
        self._path = path

    def _dump_metric_history(self) -> None:
        if len(self._full_metrics) > 0:
            metric_list = [self._full_metrics[ep] for ep in sorted(self._full_metrics.keys())]
            df = pd.DataFrame.from_records(metric_list)
            df.to_csv(os.path.join(self._path, "metrics_full.csv"), index=True)
        if len(self._valid_metrics) > 0:
            metric_list = [self._valid_metrics[ep] for ep in sorted(self._valid_metrics.keys())]
            df = pd.DataFrame.from_records(metric_list)
            df.to_csv(os.path.join(self._path, "metrics_valid.csv"), index=True)

    def on_training_end(self, ep: int) -> None:
        if len(self.env_sampler.metrics) > 0:
            metrics = copy.deepcopy(self.env_sampler.metrics)
            metrics["ep"] = ep
            if ep in self._full_metrics:
                self._full_metrics[ep].update(metrics)
            else:
                self._full_metrics[ep] = metrics
        self._dump_metric_history()

    def on_validation_end(self, ep: int) -> None:
        if len(self.env_sampler.metrics) > 0:
            metrics = copy.deepcopy(self.env_sampler.metrics)
            metrics["ep"] = ep
            if ep in self._full_metrics:
                self._full_metrics[ep].update(metrics)
            else:
                self._full_metrics[ep] = metrics
            if ep in self._valid_metrics:
                self._valid_metrics[ep].update(metrics)
            else:
                self._valid_metrics[ep] = metrics
        self._dump_metric_history()


class CallbackManager(object):
    def __init__(
        self,
        workflow: TrainingWorkflow,
        callbacks: List[Callback],
        env_sampler: EnvSampler,
        training_manager: TrainingManager,
        logger: LoggerV2,
    ) -> None:
        super(CallbackManager, self).__init__()

        self._callbacks = callbacks
        for callback in self._callbacks:
            callback.workflow = workflow
            callback.env_sampler = env_sampler
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
