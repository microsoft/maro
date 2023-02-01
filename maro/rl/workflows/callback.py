# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import copy
import os
from typing import Dict, List, Union

import pandas as pd

from maro.rl.rollout import AbsEnvSampler, BatchEnvSampler
from maro.rl.training import TrainingManager
from maro.utils import LoggerV2

EnvSampler = Union[AbsEnvSampler, BatchEnvSampler]


class Callback(object):
    def on_episode_start(
        self,
        env_sampler: EnvSampler,
        training_manager: TrainingManager,
        logger: LoggerV2,
        ep: int,
    ) -> None:
        pass

    def on_episode_end(
        self,
        env_sampler: EnvSampler,
        training_manager: TrainingManager,
        logger: LoggerV2,
        ep: int,
    ) -> None:
        pass

    def on_training_start(
        self,
        env_sampler: EnvSampler,
        training_manager: TrainingManager,
        logger: LoggerV2,
        ep: int,
    ) -> None:
        pass

    def on_training_end(
        self,
        env_sampler: EnvSampler,
        training_manager: TrainingManager,
        logger: LoggerV2,
        ep: int,
    ) -> None:
        pass

    def on_validation_start(
        self,
        env_sampler: EnvSampler,
        training_manager: TrainingManager,
        logger: LoggerV2,
        ep: int,
    ) -> None:
        pass

    def on_validation_end(
        self,
        env_sampler: EnvSampler,
        training_manager: TrainingManager,
        logger: LoggerV2,
        ep: int,
    ) -> None:
        pass

    def on_test_start(
        self,
        env_sampler: EnvSampler,
        training_manager: TrainingManager,
        logger: LoggerV2,
        ep: int,
    ) -> None:
        pass

    def on_test_end(
        self,
        env_sampler: EnvSampler,
        training_manager: TrainingManager,
        logger: LoggerV2,
        ep: int,
    ) -> None:
        pass


class Checkpoint(Callback):
    def __init__(self, path: str, interval: int) -> None:
        super(Checkpoint, self).__init__()

        self._path = path
        self._interval = interval

    def on_training_end(
        self,
        env_sampler: EnvSampler,
        training_manager: TrainingManager,
        logger: LoggerV2,
        ep: int,
    ) -> None:
        if ep % self._interval == 0:
            training_manager.save(os.path.join(self._path, str(ep)))
            logger.info(f"[Episode {ep}] All trainer states saved under {self._path}")


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

    def on_training_end(
        self,
        env_sampler: EnvSampler,
        training_manager: TrainingManager,
        logger: LoggerV2,
        ep: int,
    ) -> None:
        if len(env_sampler.metrics) > 0:
            metrics = copy.deepcopy(env_sampler.metrics)
            metrics["ep"] = ep
            if ep in self._full_metrics:
                self._full_metrics[ep].update(metrics)
            else:
                self._full_metrics[ep] = metrics
        self._dump_metric_history()

    def on_validation_end(
        self,
        env_sampler: EnvSampler,
        training_manager: TrainingManager,
        logger: LoggerV2,
        ep: int,
    ) -> None:
        if len(env_sampler.metrics) > 0:
            metrics = copy.deepcopy(env_sampler.metrics)
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


SUPPORTED_CALLBACK_FUNC = {
    "on_episode_start",
    "on_episode_end",
    "on_training_start",
    "on_training_end",
    "on_validation_start",
    "on_validation_end",
    "on_test_start",
    "on_test_end",
}


class CallbackManager(object):
    def __init__(self, callbacks: List[Callback]) -> None:
        super(CallbackManager, self).__init__()

        self._callbacks = callbacks

    def call(
        self,
        func_name: str,
        env_sampler: EnvSampler,
        training_manager: TrainingManager,
        logger: LoggerV2,
        ep: int,
    ) -> None:
        assert func_name in SUPPORTED_CALLBACK_FUNC

        for callback in self._callbacks:
            func = getattr(callback, func_name)
            func(env_sampler, training_manager, logger, ep)
