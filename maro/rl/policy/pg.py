# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch

from maro.communication import SessionMessage
from maro.rl.types import DiscretePolicyNet, Trajectory
from maro.rl.utils import MsgKey, MsgTag, discount_cumsum

from .policy import Batch, LossInfo, RLPolicy


class PGActionInfo:

    __slots__ = ["action", "logp", "value"]

    def __init__(self, action, logp: float, value: float):
        self.action = action
        self.logp = logp
        self.value = value


class PGBatch(Batch):

    __slots__ = ["states", "actions", "returns", "logps"]

    def __init__(self, states, returns: np.array):
        super().__init__()
        self.states = states
        self.returns = returns

    @property
    def size(self):
        return len(self.states)


class PGLossInfo(LossInfo):
    def __init__(self, loss, grad=None):
        super().__init__(loss, grad)


class PolicyGradient(RLPolicy):
    """The vanilla Policy Gradient (VPG) algorithm, a.k.a., REINFORCE.

    Reference: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.

    Args:
        name (str): Unique identifier for the policy.
        policy_net (DiscretePolicyNet): Multi-task model that computes action distributions and state values.
            It may or may not have a shared bottom stack.
        reward_discount (float): Reward decay as defined in standard RL terminology.
        grad_iters (int): Number of gradient steps for each batch or set of batches. Defaults to 1.
    """
    def __init__(
        self,
        name: str,
        policy_net: DiscretePolicyNet,
        reward_discount: float,
        grad_iters: int = 1,
        get_loss_on_rollout_finish: bool = False,
        remote: bool = False
    ):
        if not isinstance(policy_net, DiscretePolicyNet):
            raise TypeError("model must be an instance of 'DiscretePolicyNet'")
        super().__init__(name, remote=remote)
        self.policy_net = policy_net
        self.device = self.policy_net.device
        self.reward_discount = reward_discount
        self.grad_iters = grad_iters
        self._get_loss_on_rollout_finish = get_loss_on_rollout_finish

    def choose_action(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return actions and log probabilities for given states."""
        self.policy_net.eval()
        with torch.no_grad():
            actions, log_p = self.policy_net.get_action(states)
        actions, log_p = actions.cpu().numpy(), log_p.cpu().numpy()
        return (actions[0], log_p[0]) if len(actions) == 1 else actions, log_p

    def get_rollout_info(self, trajectory: Trajectory):
        if self._get_loss_on_rollout_finish:
            return self.get_batch_loss(self._preprocess(trajectory))
        else:
            return trajectory

    def _preprocess(self, trajectory: Trajectory) -> PGBatch:
        rewards = np.append(trajectory.rewards, trajectory.actions[-1].value if trajectory.actions[-1] else .0)
        return PGBatch(trajectory.states[:-1], discount_cumsum(rewards, self.reward_discount)[:-1])

    def get_batch_loss(self, batch: PGBatch, explicit_grad: bool = False):
        """
        This should be called at the end of a simulation episode and the experiences obtained from
        the experience store's ``get`` method should be a sequential set, i.e., in the order in
        which they are generated during the simulation. Otherwise, the return values may be meaningless.
        """
        assert self.policy_net.trainable, "policy_net needs to have at least one optimizer registered."
        self.policy_net.train()

        states = batch.states
        returns = torch.from_numpy(np.asarray(batch.returns)).to(self.device)

        _, logp = self.policy_net(states)
        loss = -(logp * returns).mean()
        grad = self.policy_net.get_gradients(loss) if explicit_grad else None
        return PGLossInfo(loss, grad=grad)

    def update_with_multi_loss_info(self, loss_info_list: List[PGLossInfo]):
        """Apply gradients to the underlying parameterized model."""
        self.policy_net.apply_gradients([loss_info.grad for loss_info in loss_info_list])

    def learn_from_multi_trajectories(self, trajectories: List[Trajectory]):
        if self.remote:
            # TODO: distributed grad computation
            pass
        else:
            batches = [self._preprocess(traj) for traj in trajectories]
            for _ in range(self.grad_iters):
                self.update_with_multi_loss_info([self.get_batch_loss(batch, explicit_grad=True) for batch in batches])

    def distributed_learn(self, rollout_info, worker_id_list):
        assert self.remote, "distributed_learn is invalid when self.remote is False!"

        batches = [self._preprocess(traj) for traj in rollout_info]
        for _ in range(self.grad_iters):
            msg_dict = defaultdict(lambda: defaultdict(dict))
            for i, worker_id in enumerate(worker_id_list):
                msg_dict[worker_id][MsgKey.GRAD_TASK][self._name] = batches[i::len(worker_id)]
                msg_dict[worker_id][MsgKey.POLICY_STATE][self._name] = self.get_state()
                # data-parallel
                self._proxy.isend(SessionMessage(
                    MsgTag.COMPUTE_GRAD, self._proxy.name, worker_id, body=msg_dict[worker_id]))
            dones = 0
            loss_infos = {self._name: []}
            for msg in self._proxy.receive():
                if msg.tag == MsgTag.COMPUTE_GRAD_DONE:
                    for policy_name, loss_info in msg.body[MsgKey.LOSS_INFO].items():
                        if isinstance(loss_info, list):
                            loss_infos[policy_name] += loss_info
                        elif isinstance(loss_info, PGLossInfo):
                            loss_infos[policy_name].append(loss_info)
                        else:
                            raise TypeError(f"Wrong type of loss_info: {type(loss_info)}")
                    dones += 1
                    if dones == len(msg_dict):
                        break
            # build dummy computation graph before apply gradients.
            _ = self.get_batch_loss(batches[0], explicit_grad=True)
            self.update_with_multi_loss_info(loss_infos[self._name])

    def set_state(self, policy_state):
        self.policy_net.load_state_dict(policy_state)

    def get_state(self):
        return self.policy_net.state_dict()