# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import dgl
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


@dataclass
class GraphBasedExpElement:
    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    is_done: np.ndarray
    graph: dgl.DGLGraph

    def split_contents_by_trainer(self, agent2trainer: Dict[Any, str]) -> Dict[str, GraphBasedExpElement]:
        """Split the ExpElement's contents by trainer.

        Args:
            agent2trainer (Dict[Any, str]): Mapping of agent name and trainer name.

        Returns:
            Contents (Dict[str, ExpElement]): A dict that contains the ExpElements of all trainers. The key of this
                dict is the trainer name.
        """
        return {trainer_name: self for trainer_name in agent2trainer.values()}


@dataclass
class GraphBasedTransitionBatch:
    states: np.ndarray
    actions: np.ndarray
    returns: np.ndarray
    graph: dgl.DGLGraph
    advantages: np.ndarray
    old_logps: np.ndarray


class BatchGraphReplayMemory:
    def __init__(
        self,
        max_t: int,
        graph_batch_size: int,
        num_samples: int,
        feature_size: int,
    ) -> None:
        self.max_t = max_t
        self.graph_batch_size: int = graph_batch_size
        self.num_nodes: int = None
        self.num_samples: int = num_samples
        self.feature_size: int = feature_size
        self._t = 0

        self.graph: dgl.DGLGraph = None
        self.states: np.ndarray = None  # shape [max_t + 1, num_nodes, num_samples, feature_size]
        self.actions: np.ndarray = None  # shape [max_t, num_nodes, num_samples]
        self.action_logps: np.ndarray = None

        array_size = (self.max_t + 1, graph_batch_size, num_samples)
        self.rewards: np.ndarray = np.zeros(array_size, dtype=np.float32)
        self.is_done: np.ndarray = np.ones(array_size, dtype=np.int8)
        self.returns: np.ndarray = np.zeros(array_size, dtype=np.float32)
        self.advantages: np.ndarray = np.zeros(array_size, dtype=np.float32)
        self.value_preds: np.ndarray = np.zeros(array_size, dtype=np.float32)

    def _init_storage(self, graph: dgl.DGLGraph) -> None:
        self.graph = graph
        self.num_nodes = graph.num_nodes()

        array_size = (self.max_t + 1, self.num_nodes, self.num_samples)
        self.states = np.zeros((*array_size, self.feature_size), dtype=np.float32)
        self.actions = np.zeros(array_size, dtype=np.float32)
        self.action_logps = np.zeros(array_size, dtype=np.float32)

    def add_transition(self, exp_element: GraphBasedExpElement, value_pred: np.ndarray, logps: np.ndarray) -> None:
        if self._t == 0:
            assert exp_element.graph is not None
            self._init_storage(exp_element.graph)

        self.states[self._t] = exp_element.state
        self.actions[self._t] = exp_element.action
        self.rewards[self._t] = exp_element.reward
        self.value_preds[self._t] = value_pred
        self.action_logps[self._t] = logps
        self.is_done[self._t + 1] = exp_element.is_done

        self._t += 1

    def build_update_sampler(self, batch_size: int, num_train_epochs: int, gamma: float) -> GraphBasedTransitionBatch:
        for t in reversed(range(self.max_t)):
            self.returns[t] = self.rewards[t] + gamma * (1 - self.is_done[t + 1]) * self.returns[t + 1]

        advantages = self.returns[:-1] - self.value_preds[:-1]
        self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        flat_states = np.transpose(self.states[: self._t], (0, 2, 1, 3)).reshape(-1, self.num_nodes, self.feature_size)
        flat_actions = np.transpose(self.actions[: self._t], (0, 2, 1)).reshape(-1, self.num_nodes)
        flat_returns = np.transpose(self.returns[: self._t], (0, 2, 1)).reshape(-1, self.graph_batch_size)
        flat_advantages = np.transpose(self.advantages[: self._t], (0, 2, 1)).reshape(-1, self.graph_batch_size)
        flat_action_logps = np.transpose(self.action_logps[: self._t], (0, 2, 1)).reshape(-1, self.num_nodes)

        flat_dim = flat_states.shape[0]

        sampler = BatchSampler(
            sampler=SubsetRandomSampler(range(flat_dim)),
            batch_size=min(flat_dim, batch_size),
            drop_last=False,
        )

        sampler_t = 0
        while sampler_t < num_train_epochs:
            for idx in sampler:
                yield GraphBasedTransitionBatch(
                    states=flat_states[idx],
                    actions=flat_actions[idx],
                    returns=flat_returns[idx],
                    graph=self.graph,
                    advantages=flat_advantages[idx],
                    old_logps=flat_action_logps[idx],
                )

                sampler_t += 1
                if sampler_t == num_train_epochs:
                    break

    def reset(self) -> None:
        self.is_done[0] = 0
        self._t = 0

    def get_statistics(self) -> Dict[str, float]:
        action_0_count = np.count_nonzero(self.actions[: self._t] == 0)
        action_1_count = np.count_nonzero(self.actions[: self._t] == 1)
        action_2_count = np.count_nonzero(self.actions[: self._t] == 2)
        action_count = action_0_count + action_1_count + action_2_count
        assert action_count == self.actions[: self._t].size
        return {
            "step_t": self._t,
            "action_0": action_0_count / action_count,
            "action_1": action_1_count / action_count,
            "action_2": action_2_count / action_count,
            "reward": self.rewards[: self._t].mean(),
            "return": self.returns[: self._t].mean(),
            "advantage": self.advantages[: self._t].mean(),
        }
