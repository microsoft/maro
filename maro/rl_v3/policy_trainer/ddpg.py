from typing import Callable, Dict, Optional

import torch

from maro.rl_v3.model import QNet
from maro.rl_v3.policy import ContinuousRLPolicy, RLPolicy
from maro.rl_v3.replay_memory import RandomReplayMemory
from maro.rl_v3.utils import TransitionBatch, ndarray_to_tensor
from maro.utils import clone

from .abs_trainer import SingleTrainer
from .train_worker import SingleTrainWorker


class DDPGWorker(SingleTrainWorker):
    def __init__(
        self,
        name: str,
        device: torch.device,
        get_q_critic_net_func: Callable[[], QNet],
        reward_discount: float,
        q_value_loss_cls: Callable = None,
        soft_update_coef: float = 1.0,
        critic_loss_coef: float = 0.1,
        enable_data_parallelism: bool = False
    ) -> None:
        super(DDPGWorker, self).__init__(name, device, enable_data_parallelism)

        self._policy: Optional[ContinuousRLPolicy] = None
        self._target_policy: Optional[ContinuousRLPolicy] = None
        self._q_critic_net: Optional[QNet] = None
        self._target_q_critic_net: Optional[QNet] = None
        self._get_q_critic_net_func = get_q_critic_net_func

        self._reward_discount = reward_discount
        self._q_value_loss_func = q_value_loss_cls() if q_value_loss_cls is not None else torch.nn.MSELoss()
        self._critic_loss_coef = critic_loss_coef
        self._soft_update_coef = soft_update_coef

    def _register_policy_impl(self, policy: RLPolicy) -> None:
        assert isinstance(policy, ContinuousRLPolicy)

        self._policy = policy
        self._target_policy = clone(self._policy)
        self._target_policy.set_name(f"target_{policy.name}")
        self._target_policy.eval()
        self._target_policy.to_device(self._device)

        self._q_critic_net = self._get_q_critic_net_func()
        self._q_critic_net.to(self._device)
        self._target_q_critic_net: QNet = clone(self._q_critic_net)
        self._target_q_critic_net.eval()
        self._target_q_critic_net.to(self._device)

    def get_batch_grad(
        self,
        batch: TransitionBatch,
        tensor_dict: Dict[str, object] = None,
        scope: str = "all"
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Reference: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
        """

        assert scope in ("all", "actor", "critic"), \
            f"Unrecognized scope {scope}. Excepting 'all', 'actor', or 'critic'."

        grad_dict = {}
        if scope in ("all", "critic"):
            grad_dict["critic_grad"] = self._get_critic_grad(batch)

        if scope in ("all", "actor"):
            grad_dict["actor_grad"] = self._get_actor_grad(batch)

        return grad_dict

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

    def get_worker_state_dict(self, scope: str = "all") -> dict:
        ret_dict = {}
        if scope in ("all", "actor"):
            ret_dict["policy_state"] = self._policy.get_policy_state()
            ret_dict["target_policy_state"] = self._target_policy.get_policy_state()
        if scope in ("all", "critic"):
            ret_dict["critic_state"] = self._q_critic_net.get_net_state()
            ret_dict["target_critic_state"] = self._target_q_critic_net.get_net_state()
        return ret_dict

    def set_worker_state_dict(self, worker_state_dict: dict, scope: str = "all") -> None:
        if scope in ("all", "actor"):
            self._policy.set_policy_state(worker_state_dict["policy_state"])
            self._target_policy.set_policy_state(worker_state_dict["target_policy_state"])
        if scope in ("all", "critic"):
            self._q_critic_net.set_net_state(worker_state_dict["critic_state"])
            self._target_q_critic_net.set_net_state(worker_state_dict["target_critic_state"])

    def soft_update_target(self) -> None:
        self._target_policy.soft_update(self._policy, self._soft_update_coef)
        self._target_q_critic_net.soft_update(self._q_critic_net, self._soft_update_coef)


class DDPG(SingleTrainer):
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:
        https://arxiv.org/pdf/1509.02971.pdf
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ddpg

    Args:
        name (str): Unique identifier for the policy.
        get_q_critic_net_func (Callable[[], QNet]): Function to get Q critic net.
        reward_discount (float): Reward decay as defined in standard RL terminology.
        q_value_loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class for
            the Q-value loss. If it is a string, it must be a key in ``TORCH_LOSS``. Defaults to "mse".
        policy (DiscretePolicyGradient): The policy to be trained.
        random_overwrite (bool): This specifies overwrite behavior when the replay memory capacity is reached. If True,
            overwrite positions will be selected randomly. Otherwise, overwrites will occur sequentially with
            wrap-around. Defaults to False.
        replay_memory_capacity (int): Capacity of the replay memory. Defaults to 10000.
        num_epochs (int): Number of training epochs per call to ``learn``. Defaults to 1.
        update_target_every (int): Number of training rounds between policy target model updates.
        soft_update_coef (float): Soft update coefficient, e.g., target_model = (soft_update_coef) * eval_model +
            (1-soft_update_coef) * target_model. Defaults to 1.0.
        train_batch_size (int): Batch size for training the Q-net. Defaults to 32.
        critic_loss_coef (float): Coefficient for critic loss in total loss. Defaults to 1.0.
        device (str): Identifier for the torch device. The policy will be moved to the specified device. If it is
            None, the device will be set to "cpu" if cuda is unavailable and "cuda" otherwise. Defaults to None.
        enable_data_parallelism (bool): Whether to enable data parallelism in this trainer. Defaults to False.
    """

    def __init__(
        self,
        name: str,
        get_q_critic_net_func: Callable[[], QNet],
        reward_discount: float,
        q_value_loss_cls: Callable = None,
        policy: ContinuousRLPolicy = None,
        random_overwrite: bool = False,
        replay_memory_capacity: int = 10000,
        num_epochs: int = 1,
        update_target_every: int = 5,
        soft_update_coef: float = 1.0,
        train_batch_size: int = 32,
        critic_loss_coef: float = 0.1,
        device: str = None,
        enable_data_parallelism: bool = False
    ) -> None:
        super(DDPG, self).__init__(
            name=name,
            device=device,
            enable_data_parallelism=enable_data_parallelism,
            train_batch_size=train_batch_size
        )

        self._get_q_critic_net_func = get_q_critic_net_func

        self._replay_memory_capacity = replay_memory_capacity
        self._random_overwrite = random_overwrite
        if policy is not None:
            self.register_policy(policy)

        self._num_epochs = num_epochs
        self._policy_version = self._target_policy_version = 0
        self._update_target_every = update_target_every
        self._soft_update_coef = soft_update_coef
        self._train_batch_size = train_batch_size
        self._reward_discount = reward_discount

        self._critic_loss_coef = critic_loss_coef
        self._q_value_loss_cls = q_value_loss_cls

    def _register_policy_impl(self, policy: ContinuousRLPolicy) -> None:
        self._worker = DDPGWorker(
            name="worker", device=self._device, get_q_critic_net_func=self._get_q_critic_net_func,
            reward_discount=self._reward_discount, q_value_loss_cls=self._q_value_loss_cls,
            soft_update_coef=self._soft_update_coef, critic_loss_coef=self._critic_loss_coef,
            enable_data_parallelism=self._enable_data_parallelism
        )
        self._worker.register_policy(policy)

        self._replay_memory = RandomReplayMemory(
            capacity=self._replay_memory_capacity, state_dim=policy.state_dim,
            action_dim=policy.action_dim, random_overwrite=self._random_overwrite
        )

    def _train_step_impl(self) -> None:
        for _ in range(self._num_epochs):
            self._worker.set_batch(self._get_batch())
            self._worker.update()
            self._try_soft_update_target()

    def _try_soft_update_target(self) -> None:
        self._policy_version += 1
        if self._policy_version - self._target_policy_version == self._update_target_every:
            self._worker.soft_update_target()
            self._target_policy_version = self._policy_version

    def get_policy_state(self) -> object:
        return self._worker.get_policy_state()

    def set_policy_state(self, policy_state: object) -> None:
        self._worker.set_policy_state(policy_state)
