# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import List, Union

import numpy as np
import torch

from maro.communication import SessionMessage
from maro.rl.exploration import GaussianNoiseExploration, NoiseExploration
from maro.rl.types import ContinuousACNet, Trajectory
from maro.rl.utils import MsgKey, MsgTag, get_torch_loss_cls
from maro.utils.exception.rl_toolkit_exception import InvalidExperience

from .policy import Batch, LossInfo, RLPolicy
from .replay import ReplayMemory


class DDPGBatch(Batch):
    """Wrapper for a set of experiences.

    An experience consists of state, action, reward, next state and auxillary information.
    """
    __slots__ = ["states", "actions", "rewards", "next_states"]

    def __init__(self, states: list, actions: list, rewards: list, next_states: list):
        if not len(states) == len(actions) == len(rewards) == len(next_states):
            raise InvalidExperience("values of contents should consist of lists of the same length")
        super().__init__()
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states

    @property
    def size(self):
        return len(self.states)


class DDPGLossInfo(LossInfo):

    __slots__ = ["policy_loss", "q_loss"]

    def __init__(self, loss, policy_loss, q_loss, grad=None):
        super().__init__(loss, grad)
        self.loss = loss
        self.policy_loss = policy_loss
        self.q_loss = q_loss
        self.grad = grad


class DDPG(RLPolicy):
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:
        https://arxiv.org/pdf/1509.02971.pdf
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ddpg

    Args:
        name (str): Unique identifier for the policy.
        ac_net (ContinuousACNet): DDPG policy and q-value models.
        reward_discount (float): Reward decay as defined in standard RL terminology.
        update_target_every (int): Number of training rounds between policy target model updates.
        q_value_loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class for
            the Q-value loss. If it is a string, it must be a key in ``TORCH_LOSS``. Defaults to "mse".
        q_value_loss_coeff (float): Coefficient for policy loss in the total loss function, e.g.,
            loss = policy_loss + ``q_value_loss_coeff`` * q_value_loss. Defaults to 1.0.
        soft_update_coeff (float): Soft update coefficient, e.g., target_model = (soft_update_coeff) * eval_model +
            (1-soft_update_coeff) * target_model. Defaults to 1.0.
        exploration: Exploration strategy for generating exploratory actions. Defaults to ``GaussianNoiseExploration``.
        replay_memory_capacity (int): Capacity of the replay memory. Defaults to 10000.
        random_overwrite (bool): This specifies overwrite behavior when the replay memory capacity is reached. If True,
            overwrite positions will be selected randomly. Otherwise, overwrites will occur sequentially with
            wrap-around. Defaults to False.
    """
    def __init__(
        self,
        name: str,
        ac_net: ContinuousACNet,
        reward_discount: float,
        num_epochs: int = 1,
        update_target_every: int = 5,
        q_value_loss_cls="mse",
        q_value_loss_coeff: float = 1.0,
        soft_update_coeff: float = 1.0,
        exploration: NoiseExploration = GaussianNoiseExploration(),
        replay_memory_capacity: int = 10000,
        random_overwrite: bool = False,
        remote: bool = False
    ):
        if not isinstance(ac_net, ContinuousACNet):
            raise TypeError("model must be an instance of 'ContinuousACNet'")

        super().__init__(name, remote=remote)
        self.ac_net = ac_net
        self.device = self.ac_net.device
        if self.ac_net.trainable:
            self.target_ac_net = ac_net.copy()
            self.target_ac_net.eval()
        else:
            self.target_ac_net = None
        self.reward_discount = reward_discount
        self.num_epochs = num_epochs
        self.update_target_every = update_target_every
        self.q_value_loss_func = get_torch_loss_cls(q_value_loss_cls)()
        self.q_value_loss_coeff = q_value_loss_coeff
        self.soft_update_coeff = soft_update_coeff

        self._ac_net_version = 0
        self._target_ac_net_version = 0

        self._replay_memory = ReplayMemory(DDPGBatch, replay_memory_capacity, random_overwrite=random_overwrite)

        self.exploration = exploration
        self.exploring = True  # set initial exploration status to True

    def choose_action(self, states, explore: bool = False) -> Union[float, np.ndarray]:
        self.ac_net.eval()
        with torch.no_grad():
            actions = self.ac_net.get_action(states).cpu().numpy()

        if explore:
            actions = self.exploration(actions, state=states)
        return actions[0] if len(actions) == 1 else actions

    def _preprocess(self, trajectory: Trajectory):
        if trajectory.states[-1]:
            batch = DDPGBatch(
                states=trajectory.states[:-1],
                actions=trajectory.actions[:-1],
                rewards=trajectory.rewards,
                next_states=trajectory.states[1:]
            )
        else:
            batch = DDPGBatch(
                states=trajectory.states[:-2],
                actions=trajectory.actions[:-2],
                rewards=trajectory.rewards[:-1],
                next_states=trajectory.states[1:-1]
            )
        self._replay_memory.put(batch)

    def _get_batch(self) -> DDPGBatch:
        indexes = np.random.choice(self._replay_memory.size)
        return DDPGBatch(
            [self._replay_memory.data["states"][idx] for idx in indexes],
            [self._replay_memory.data["actions"][idx] for idx in indexes],
            [self._replay_memory.data["rewards"][idx] for idx in indexes],
            [self._replay_memory.data["next_states"][idx] for idx in indexes]
        )

    def get_batch_loss(self, batch: DDPGBatch, explicit_grad: bool = False) -> DDPGLossInfo:
        assert self.ac_net.trainable, "ac_net needs to have at least one optimizer registered."
        self.ac_net.train()
        states, next_states = batch.states, batch.next_states
        actual_actions = torch.from_numpy(batch.actions).to(self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.device)
        if len(actual_actions.shape) == 1:
            actual_actions = actual_actions.unsqueeze(dim=1)  # (N, 1)

        with torch.no_grad():
            next_q_values = self.target_ac_net.value(next_states)
        target_q_values = (rewards + self.reward_discount * next_q_values).detach()  # (N,)

        q_values = self.ac_net(states, actions=actual_actions).squeeze(dim=1)  # (N,)
        q_loss = self.q_value_loss_func(q_values, target_q_values)
        policy_loss = -self.ac_net.value(states).mean()

        # total loss
        loss = policy_loss + self.q_value_loss_coeff * q_loss
        grad = self.ac_net.get_gradients(loss) if explicit_grad else None
        return DDPGLossInfo(policy_loss, q_loss, loss, grad=grad)

    def update_with_multi_loss_info(self, loss_info_list: List[DDPGLossInfo]):
        self.ac_net.apply_gradients([loss_info.grad for loss_info in loss_info_list])
        if self._ac_net_version - self._target_ac_net_version == self.update_target_every:
            self._update_target()

    def learn_from_multi_trajectories(self, trajectories: List[Trajectory]):
        for traj in trajectories:
            self._preprocess(traj)

        if self.remote:
            # TODO: distributed grad computation
            pass
        else:
            for _ in range(self.num_epochs):
                loss_info = self.get_batch_loss(self._get_batch(), explicit_grad=False)
                self.ac_net.step(loss_info.loss)
                self._ac_net_version += 1
                if self._ac_net_version - self._target_ac_net_version == self.update_target_every:
                    self._update_target()

    def distributed_learn(self, rollout_info, worker_id_list):
        assert self.remote, "distributed_learn is invalid when self.remote is False!"

        for traj in rollout_info:
            self._preprocess(traj)

        for _ in range(self.num_epochs):
            msg_dict = defaultdict(lambda: defaultdict(dict))
            for worker_id in worker_id_list:
                msg_dict[worker_id][MsgKey.GRAD_TASK][self._name] = self._get_batch()
                msg_dict[worker_id][MsgKey.POLICY_STATE][self._name] = self.get_state()
                # data-parallel by multiple hosts/workers
                self._proxy.isend(SessionMessage(
                    MsgTag.COMPUTE_GRAD, self._proxy.name, worker_id, body=msg_dict[worker_id]))
            dones = 0
            loss_infos = {self._name: []}
            for msg in self._proxy.receive():
                if msg.tag == MsgTag.COMPUTE_GRAD_DONE:
                    for policy_name, loss_info in msg.body[MsgKey.LOSS_INFO].items():
                        if isinstance(loss_info, list):
                            loss_infos[policy_name] += loss_info
                        elif isinstance(loss_info, DDPGLossInfo):
                            loss_infos[policy_name].append(loss_info)
                        else:
                            raise TypeError(f"Wrong type of loss_info: {type(loss_info)}")
                    dones += 1
                    if dones == len(msg_dict):
                        break
            # build dummy computation graph before apply gradients.
            _ = self.get_batch_loss(self._get_batch(), explicit_grad=True)
            self.update_with_multi_loss_info(loss_infos[self._name])

    def _update_target(self):
        # soft-update target network
        self.target_ac_net.soft_update(self.ac_net, self.soft_update_coeff)
        self._target_ac_net_version = self._ac_net_version

    @property
    def exploration_params(self):
        return self.exploration.parameters

    def exploit(self):
        self.exploring = False

    def explore(self):
        self.exploring = True

    def exploration_step(self):
        self.exploration.step()

    def set_state(self, policy_state):
        self.ac_net.load_state_dict(policy_state)
        self.target_ac_net = self.ac_net.copy() if self.ac_net.trainable else None

    def get_state(self):
        return self.ac_net.state_dict()
