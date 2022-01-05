# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# TODO: DDPG has net been tested in a real test case

from dataclasses import dataclass
from typing import Callable, Dict, List

import torch

from maro.rl_v3.model import QNet
from maro.rl_v3.policy import ContinuousRLPolicy
from maro.rl_v3.training import AbsTrainOps, RandomReplayMemory, SingleTrainer, TrainerParams
from maro.rl_v3.utils import CoroutineWrapper, TransitionBatch, ndarray_to_tensor
from maro.utils import clone


@dataclass
class DDPGParams(TrainerParams):
    reward_discount: float = 0.9
    num_epochs: int = 1
    update_target_every: int = 5
    q_value_loss_cls: Callable = None
    soft_update_coef: float = 1.0
    critic_loss_coef: float = 0.1
    random_overwrite: bool = False


class DDPGOps(AbsTrainOps):
    def __init__(
        self,
        device: str,
        get_policy_func: Callable[[], ContinuousRLPolicy],
        get_q_critic_net_func: Callable[[], QNet],
        enable_data_parallelism: bool = False,
        *,
        reward_discount: float,
        q_value_loss_cls: Callable = None,
        soft_update_coef: float = 1.0,
        critic_loss_coef: float = 0.1
    ) -> None:
        super(DDPGOps, self).__init__(
            device=device,
            is_single_scenario=True,
            get_policy_func=get_policy_func,
            enable_data_parallelism=enable_data_parallelism
        )

        assert isinstance(self._policy, ContinuousRLPolicy)

        self._target_policy = clone(self._policy)
        self._target_policy.set_name(f"target_{self._policy.name}")
        self._target_policy.eval()
        self._target_policy.to_device(self._device)
        self._q_critic_net = get_q_critic_net_func()
        self._q_critic_net.to(self._device)
        self._target_q_critic_net: QNet = clone(self._q_critic_net)
        self._target_q_critic_net.eval()
        self._target_q_critic_net.to(self._device)

        self._reward_discount = reward_discount
        self._q_value_loss_func = q_value_loss_cls() if q_value_loss_cls is not None else torch.nn.MSELoss()
        self._critic_loss_coef = critic_loss_coef
        self._soft_update_coef = soft_update_coef

    def get_batch_grad(
        self,
        batch: TransitionBatch,
        tensor_dict: Dict[str, object] = None,
        scope: str = "all"
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Reference: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
        """

        assert scope in ("all", "actor", "critic"), \
            f"Unrecognized scope {scope}. Excepting 'all', 'actor', or 'critic'."

        grad_dict = {}
        if scope in ("all", "critic"):
            grad_dict["critic_grad"] = self._get_critic_grad(batch)

        if scope in ("all", "actor"):
            grad_dict["actor_grad"] = self._get_actor_grad(batch)

        return grad_dict

    def _dispatch_batch(self, batch: TransitionBatch, num_ops: int) -> List[TransitionBatch]:
        raise NotImplementedError

    def _dispatch_tensor_dict(self, tensor_dict: Dict[str, object], num_ops: int) -> List[Dict[str, object]]:
        raise NotImplementedError

    def _get_critic_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        self._q_critic_net.train()
        self._policy.train()

        states = ndarray_to_tensor(batch.states, self._device)  # s

        policy_loss = -self._q_critic_net.q_values(
            states=states,  # s
            actions=self._policy.get_actions_tensor(states)  # miu(s)
        ).mean()  # -Q(s, miu(s))

        return self._policy.get_gradients(policy_loss)

    def _get_actor_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        self._q_critic_net.train()
        self._policy.train()

        states = ndarray_to_tensor(batch.states, self._device)  # s

        next_states = ndarray_to_tensor(batch.next_states, self._device)  # s'
        actions = ndarray_to_tensor(batch.actions, self._device)  # a
        rewards = ndarray_to_tensor(batch.rewards, self._device)  # r
        terminals = ndarray_to_tensor(batch.terminals, self._device)  # d

        with torch.no_grad():
            next_q_values = self._target_q_critic_net.q_values(
                states=next_states,  # s'
                actions=self._target_policy.get_actions_tensor(next_states)  # miu_targ(s')
            )  # Q_targ(s', miu_targ(s'))

        # y(r, s', d) = r + gamma * (1 - d) * Q_targ(s', miu_targ(s'))
        target_q_values = (rewards + self._reward_discount * (1 - terminals) * next_q_values).detach()

        q_values = self._q_critic_net.q_values(states=states, actions=actions)  # Q(s, a)
        critic_loss = self._q_value_loss_func(q_values, target_q_values)  # MSE(Q(s, a), y(r, s', d))

        return self._q_critic_net.get_gradients(critic_loss * self._critic_loss_coef)

    def update(self) -> None:
        grad_dict = self._get_batch_grad(self._batch, scope="critic")
        self._q_critic_net.train()
        self._q_critic_net.apply_gradients(grad_dict["critic_grad"])

        grad_dict = self._get_batch_grad(self._batch, scope="actor")
        self._policy.train()
        self._policy.apply_gradients(grad_dict["actor_grad"])

    def get_state_dict(self, scope: str = "all") -> dict:
        ret_dict = {}
        if scope in ("all", "actor"):
            ret_dict["policy_state"] = self._policy.get_policy_state()
            ret_dict["target_policy_state"] = self._target_policy.get_policy_state()
        if scope in ("all", "critic"):
            ret_dict["critic_state"] = self._q_critic_net.get_net_state()
            ret_dict["target_critic_state"] = self._target_q_critic_net.get_net_state()
        return ret_dict

    def set_state_dict(self, ops_state_dict: dict, scope: str = "all") -> None:
        if scope in ("all", "actor"):
            self._policy.set_state(ops_state_dict["policy_state"])
            self._target_policy.set_state(ops_state_dict["target_policy_state"])
        if scope in ("all", "critic"):
            self._q_critic_net.set_net_state(ops_state_dict["critic_state"])
            self._target_q_critic_net.set_net_state(ops_state_dict["target_critic_state"])

    def soft_update_target(self) -> None:
        self._target_policy.soft_update(self._policy, self._soft_update_coef)
        self._target_q_critic_net.soft_update(self._q_critic_net, self._soft_update_coef)


class DDPG(SingleTrainer):
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:
        https://arxiv.org/pdf/1509.02971.pdf
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ddpg

    Args:
        name (str): Unique identifier for the trainer.
        policy_creator (Dict[str, Callable[[str], DiscretePolicyGradient]]): Dict of functions that used to
            create policies.
        device (str): Identifier for the torch device. The policy will be moved to the specified device. If it is
            None, the device will be set to "cpu" if cuda is unavailable and "cuda" otherwise. Defaults to None.
        enable_data_parallelism (bool): Whether to enable data parallelism in this trainer. Defaults to False.
        replay_memory_capacity (int): Capacity of the replay memory. Defaults to 10000.
        random_overwrite (bool): This specifies overwrite behavior when the replay memory capacity is reached. If True,
            overwrite positions will be selected randomly. Otherwise, overwrites will occur sequentially with
            wrap-around. Defaults to False.
        num_epochs (int): Number of training epochs per call to ``learn``. Defaults to 1.
        update_target_every (int): Number of training rounds between policy target model updates.

        reward_discount (float): Reward decay as defined in standard RL terminology.
        q_value_loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class for
            the Q-value loss. If it is a string, it must be a key in ``TORCH_LOSS``. Defaults to "mse".
        soft_update_coef (float): Soft update coefficient, e.g., target_model = (soft_update_coef) * eval_model +
            (1-soft_update_coef) * target_model. Defaults to 1.0.
        critic_loss_coef (float): Coefficient for critic loss in total loss. Defaults to 0.1.
    """

    def __init__(self, name: str, params: DDPGParams) -> None:
        super(DDPG, self).__init__(name, params)
        self._params = params
        self._policy_version = self._target_policy_version = 0
        self._q_net_version = self._target_q_net_version = 0
        self._ops_name = f"{self._name}.ops"

    def build(self) -> None:
        self._ops_params = {
            "get_policy_func": self._get_policy_func,
            **self._params.extract_ops_params(),
        }

        self._ops = self.get_ops(self._ops_name)
        self._replay_memory = RandomReplayMemory(
            capacity=self._params.replay_memory_capacity,
            state_dim=self._ops.policy_state_dim,
            action_dim=self._ops.policy_action_dim,
            random_overwrite=self._params.random_overwrite
        )

    def _get_local_ops_by_name(self, ops_name: str) -> AbsTrainOps:
        return DDPGOps(**self._ops_params)

    async def train_step(self) -> None:
        for _ in range(self._params.num_epochs):
            await self._ops.set_batch(self._get_batch())
            await self._ops.update()
            self._policy_version += 1
            if self._policy_version - self._target_policy_version == self._params.update_target_every:
                await self._ops.soft_update_target()
                self._target_policy_version = self._policy_version
