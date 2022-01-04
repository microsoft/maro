# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Callable, Dict, List

import torch

from maro.rl_v3.policy import ValueBasedPolicy
from maro.rl_v3.utils import TransitionBatch, ndarray_to_tensor
from maro.rl_v3.utils.distributed import CoroutineWrapper
from maro.utils import clone

from ..abs_train_ops import AbsTrainOps
from ..trainer import SingleTrainer, TrainerParams
from ..replay_memory import RandomReplayMemory


@dataclass
class DQNParams(TrainerParams):
    reward_discount: float = 0.9
    num_epochs: int = 1
    update_target_every: int = 5
    soft_update_coef: float = 0.1
    double: bool = False
    random_overwrite: bool = False


class DQNOps(AbsTrainOps):
    def __init__(
        self,
        device: str,
        get_policy_func: Callable[[], ValueBasedPolicy],
        enable_data_parallelism: bool = False,
        *,
        reward_discount: float = 0.9,
        soft_update_coef: float = 0.1,
        double: bool = False,
    ) -> None:
        super(DQNOps, self).__init__(
            device=device,
            is_single_scenario=True,
            get_policy_func=get_policy_func,
            enable_data_parallelism=enable_data_parallelism
        )

        assert isinstance(self._policy, ValueBasedPolicy)

        self._reward_discount = reward_discount
        self._soft_update_coef = soft_update_coef
        self._double = double
        self._loss_func = torch.nn.MSELoss()

        self._target_policy: ValueBasedPolicy = clone(self._policy)
        self._target_policy.set_name(f"target_{self._policy.name}")
        self._target_policy.eval()
        self._target_policy.to_device(self._device)

    def get_batch_grad(
        self,
        batch: TransitionBatch,
        tensor_dict: Dict[str, object] = None,
        scope: str = "all"
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        assert scope == "all", f"Unrecognized scope {scope}. Excepting 'all'."

        self._policy.train()
        states = ndarray_to_tensor(batch.states, self._device)
        next_states = ndarray_to_tensor(batch.next_states, self._device)
        actions = ndarray_to_tensor(batch.actions, self._device)
        rewards = ndarray_to_tensor(batch.rewards, self._device)
        terminals = ndarray_to_tensor(batch.terminals, self._device).float()

        with torch.no_grad():
            if self._double:
                self._policy.exploit()
                actions_by_eval_policy = self._policy.get_actions_tensor(next_states)
                next_q_values = self._target_policy.q_values_tensor(next_states, actions_by_eval_policy)
            else:
                self._target_policy.exploit()
                actions = self._target_policy.get_actions_tensor(next_states)
                next_q_values = self._target_policy.q_values_tensor(next_states, actions)
        target_q_values = (rewards + self._reward_discount * (1 - terminals) * next_q_values).detach()

        q_values = self._policy.q_values_tensor(states, actions)
        loss: torch.Tensor = self._loss_func(q_values, target_q_values)

        return {"grad": self._policy.get_gradients(loss)}

    def _dispatch_batch(self, batch: TransitionBatch, num_ops: int) -> List[TransitionBatch]:
        raise NotImplementedError

    def _dispatch_tensor_dict(self, tensor_dict: Dict[str, object], num_ops: int) -> List[Dict[str, object]]:
        raise NotImplementedError

    def get_state_dict(self, scope: str = "all") -> dict:
        return {
            "policy_state": self._policy.get_policy_state(),
            "target_q_net_state": self._target_policy.get_policy_state()
        }

    def set_state_dict(self, ops_state_dict: dict, scope: str = "all") -> None:
        self._policy.set_policy_state(ops_state_dict["policy_state"])
        self._target_policy.set_policy_state(ops_state_dict["target_q_net_state"])

    def update(self) -> None:
        grad_dict = self._get_batch_grad(self._batch)

        self._policy.train()
        self._policy.apply_gradients(grad_dict["grad"])

    def soft_update_target(self) -> None:
        self._target_policy.soft_update(self._policy, self._soft_update_coef)


class DQN(SingleTrainer):
    """The Deep-Q-Networks algorithm.

    See https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf for details.

    Args:
        name (str): Unique identifier for the trainer.
        get_policy_func_dict (Dict[str, Callable[[str], DiscretePolicyGradient]]): Dict of functions that used to
            create policies.
        device (str): Identifier for the torch device. The policy will be moved to the specified device. If it is
            None, the device will be set to "cpu" if cuda is unavailable and "cuda" otherwise. Defaults to None.
        enable_data_parallelism (bool): Whether to enable data parallelism in this trainer. Defaults to False.
        num_epochs (int): Number of training epochs per call to ``learn``. Defaults to 1.
        update_target_every (int): Number of gradient steps between target model updates. Defaults to 5.
        replay_memory_capacity (int): Capacity of the replay memory. Defaults to 10000.
        random_overwrite (bool): This specifies overwrite behavior when the replay memory capacity is reached. If True,
            overwrite positions will be selected randomly. Otherwise, overwrites will occur sequentially with
            wrap-around. Defaults to False.

        reward_discount (float): Reward decay as defined in standard RL terminology. Defaults to 0.9.
        soft_update_coef (float): Soft update coefficient, e.g.,
            target_model = (soft_update_coef) * eval_model + (1-soft_update_coef) * target_model.
            Defaults to 0.1.
        double (bool): If True, the next Q values will be computed according to the double DQN algorithm,
            i.e., q_next = Q_target(s, argmax(Q_eval(s, a))). Otherwise, q_next = max(Q_target(s, a)).
            See https://arxiv.org/pdf/1509.06461.pdf for details. Defaults to False.
    """
    def __init__(
        self,
        name: str,
        get_policy_func_dict: Dict[str, Callable[[str], ValueBasedPolicy]],
        params: DQNParams,
        device: str = None,
        enable_data_parallelism: bool = False
    ) -> None:
        super(DQN, self).__init__(name, get_policy_func_dict, params)

        self._num_epochs = params.num_epochs
        self._update_target_every = params.update_target_every

        self._replay_memory_capacity = params.replay_memory_capacity
        self._random_overwrite = params.random_overwrite

        self._q_net_version = self._target_q_net_version = 0

        self._ops_params = {
            "device": device,
            "get_policy_func": self._get_policy_func,
            "enable_data_parallelism": enable_data_parallelism,
            "reward_discount": params.reward_discount,
            "soft_update_coef": params.soft_update_coef,
            "double": params.double,
        }

    def train_step(self) -> None:
        for _ in range(self._num_epochs):
            self._ops.set_batch(self._get_batch())
            self._ops.update()

        self._q_net_version += 1
        if self._q_net_version - self._target_q_net_version == self._update_target_every:
            self._ops.soft_update_target()
            self._target_q_net_version = self._q_net_version

    def create_local_ops(self, name: str = None) -> Dict[str, Callable[[str], AbsTrainOps]]:
        return DQNOps(**self._ops_params)

    def build(self) -> None:
        self._ops = CoroutineWrapper(self.create_local_ops())
        self._replay_memory = RandomReplayMemory(
            capacity=self._replay_memory_capacity,
            state_dim=self._ops.policy_state_dim,
            action_dim=self._ops.policy_action_dim,
            random_overwrite=self._random_overwrite
        )

    async def train_step(self) -> None:
        for _ in range(self._num_epochs):
            await self._ops.set_batch(self._get_batch())
            await self._ops.update()
        self._q_net_version += 1
        if self._q_net_version - self._target_q_net_version == self._update_target_every:
            await self._ops.soft_update_target()
            self._target_q_net_version = self._q_net_version
