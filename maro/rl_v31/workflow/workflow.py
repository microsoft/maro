import os
from typing import Optional, Tuple

import pandas as pd

from maro.rl_v31.rl_component_bundle.rl_component_bundle import RLComponentBundle
from maro.rl_v31.rollout.collector import Collector
from maro.rl_v31.rollout.venv import BaseVectorEnv
from maro.rl_v31.rollout.worker import DummyEnvWorker
from maro.rl_v31.rollout.wrapper import AgentWrapper
from maro.rl_v31.training.training_manager import TrainingManager
from maro.rl_v31.workflow.callback import CallbackManager, Checkpoint, EarlyStopping, MetricsRecorder
from maro.utils import LoggerV2


class Workflow(object):
    def __init__(
        self,
        rl_component_bundle: RLComponentBundle,
        log_path: str,
        rollout_parallelism: int = 1,
        log_level_stdout: str = "INFO",
        log_level_file: str = "DEBUG",
    ) -> None:
        self._rcb = rl_component_bundle
        self._rollout_parallelism = rollout_parallelism

        self.log_path = log_path
        self.logger = LoggerV2(
            "MAIN",
            dump_path=os.path.join(log_path, "log.txt"),
            dump_mode="w",
            stdout_level=log_level_stdout,
            file_level=log_level_file,
        )

        self.train_metrics = {}
        self.valid_metrics = {}

        self.early_stop = False

        for policy_name, device in self._rcb.policy_device_mapping.items():
            policy = self._rcb.policy_dict[policy_name]
            policy.to_device(device)

    def train(
        self,
        num_iterations: int,
        steps_per_iteration: Optional[int] = None,
        episodes_per_iteration: Optional[int] = None,
        valid_steps_per_iteration: Optional[int] = None,
        valid_episodes_per_iteration: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        validation_interval: Optional[int] = None,
        explore_in_training: bool = True,
        explore_in_validation: bool = False,
        early_stop_config: Optional[Tuple[str, int, bool]] = None,
        load_path: Optional[str] = None,
    ) -> None:
        agent_wrapper = AgentWrapper(
            policy_dict={policy.name: policy for policy in self._rcb.policies},
            agent2policy=self._rcb.agent2policy,
        )
        train_collector = Collector(
            venv=BaseVectorEnv(
                env_fns=[self._rcb.env_wrapper_func for _ in range(self._rollout_parallelism)],
                worker_fn=lambda env_wrapper: DummyEnvWorker(env_wrapper),
            ),
            agent_wrapper=agent_wrapper,
        )
        valid_collector = Collector(
            venv=BaseVectorEnv(
                env_fns=[self._rcb.env_wrapper_func for _ in range(self._rollout_parallelism)],
                worker_fn=lambda env_wrapper: DummyEnvWorker(env_wrapper),
            ),
            agent_wrapper=agent_wrapper,
        )
        train_collector.reset()
        valid_collector.reset()

        training_manager = TrainingManager(
            trainers=self._rcb.trainers,
            policies=self._rcb.policies,
            agent2policy=self._rcb.agent2policy,
            policy2trainer=self._rcb.policy2trainer,
            device_mapping=self._rcb.trainer_device_mapping,
        )
        if load_path is not None:
            agent_wrapper.load(load_path)
            training_manager.load(load_path)

        # Build callbacks
        callbacks = [MetricsRecorder(path=self.log_path)]

        if checkpoint_path is not None:
            cb_checkpoint = Checkpoint(path=checkpoint_path, interval=validation_interval)
            callbacks.append(cb_checkpoint)
        else:
            cb_checkpoint = None

        if early_stop_config is not None:
            monitor, patience, higher_better = early_stop_config
            cb_early_stopping = EarlyStopping(patience=patience, monitor=monitor, higher_better=higher_better)
            callbacks.append(cb_early_stopping)
        else:
            cb_early_stopping = None

        cbm = CallbackManager(
            workflow=self,
            callbacks=callbacks,
            agent_wrapper=agent_wrapper,
            training_manager=training_manager,
            logger=self.logger,
        )

        self.early_stop = False
        for ep in range(1, num_iterations + 1):
            if self.early_stop:  # Might be set in `cbm.on_validation_end()`
                break

            self.logger.info(f"Running iteration {ep}")

            cbm.on_episode_start(ep)

            train_collector.switch_explore(explore_in_training)
            total_info, total_exps = train_collector.collect(
                n_steps=steps_per_iteration,
                n_episodes=episodes_per_iteration,
            )
            self.train_metrics = self._rcb.metrics_agg_func(total_info)

            self.logger.info(
                f"Rollout of EP {ep} finished, "
                f"collected {sum(map(len, total_exps.values()))} experiences. Start training.",
            )

            cbm.on_training_start(ep)
            training_manager.record_exp(total_exps)
            training_manager.train_step()
            cbm.on_training_end(ep)

            self.logger.info(f"Training of EP {ep} finished.")

            if validation_interval is not None and ep % validation_interval == 0:
                self.logger.info(f"Validation of EP {ep}")
                cbm.on_validation_start(ep)

                valid_collector.switch_explore(explore_in_validation)
                total_info, total_exps = valid_collector.collect(
                    n_steps=valid_steps_per_iteration,
                    n_episodes=valid_episodes_per_iteration,
                )
                self.valid_metrics = self._rcb.metrics_agg_func(total_info)

                cbm.on_validation_end(ep)

            cbm.on_episode_end(ep)

        if cb_checkpoint is not None and cb_early_stopping is not None:
            cb_checkpoint.make_copy(str(cb_early_stopping.best_ep), "best")

    def test(
        self,
        save_path: str,
        steps_per_iteration: Optional[int] = None,
        episodes_per_iteration: Optional[int] = None,
        explore: bool = False,
        load_path: Optional[str] = None,
    ) -> None:
        agent_wrapper = AgentWrapper(
            policy_dict={policy.name: policy for policy in self._rcb.policies},
            agent2policy=self._rcb.agent2policy,
        )
        collector = Collector(
            venv=BaseVectorEnv(
                env_fns=[self._rcb.env_wrapper_func for _ in range(self._rollout_parallelism)],
                worker_fn=lambda env_wrapper: DummyEnvWorker(env_wrapper),
            ),
            agent_wrapper=agent_wrapper,
        )
        collector.reset()

        if load_path is not None:
            agent_wrapper.load(load_path)

        collector.switch_explore(explore)
        total_info, total_exps = collector.collect(
            n_steps=steps_per_iteration,
            n_episodes=episodes_per_iteration,
        )
        metrics = self._rcb.metrics_agg_func(total_info)

        df = pd.DataFrame.from_dict({k: [v] for k, v in metrics.items()})
        path = os.path.join(save_path, "test_result.csv")
        self.logger.info(f"Test result dumped to {path}")
        df.to_csv(path)
