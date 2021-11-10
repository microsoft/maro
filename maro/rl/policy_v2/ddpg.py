from typing import Callable, List, Tuple, Union

import numpy as np
import torch

from maro.rl.exploration import gaussian_noise
from maro.rl.modeling_v2.ac_network import ContinuousQActorCriticNet
from maro.rl.policy_v2.policy_base import SingleRLPolicy
from maro.rl.policy_v2.policy_interfaces import ContinuousActionMixin, QNetworkMixin
from maro.rl.policy_v2.replay import ReplayMemory
from maro.rl.utils import average_grads
from maro.utils import clone


class DDPG(QNetworkMixin, ContinuousActionMixin, SingleRLPolicy):
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:
        https://arxiv.org/pdf/1509.02971.pdf
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ddpg

    Args:
        name (str): Unique identifier for the policy.
        ac_net (ContinuousACNet): DDPG policy and q-value models.
        reward_discount (float): Reward decay as defined in standard RL terminology.
        num_epochs (int): Number of training epochs per call to ``learn``. Defaults to 1.
        update_target_every (int): Number of training rounds between policy target model updates.
        q_value_loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class for
            the Q-value loss. If it is a string, it must be a key in ``TORCH_LOSS``. Defaults to "mse".
        q_value_loss_coef (float): Coefficient for policy loss in the total loss function, e.g.,
            loss = policy_loss + ``q_value_loss_coeff`` * q_value_loss. Defaults to 1.0.
        soft_update_coef (float): Soft update coefficient, e.g., target_model = (soft_update_coeff) * eval_model +
            (1-soft_update_coeff) * target_model. Defaults to 1.0.
        exploration_strategy (Tuple[Callable, dict]): A 2-tuple that consists of a) a function that takes a state
            (single or batch), an action (single or batch), the total number of possible actions and a set of keyword
            arguments, and returns an exploratory action (single or batch depending on the input), and b) a dictionary
            of keyword arguments for the function in a) (this will be assigned to the ``exploration_params`` member
            variable). Defaults to (``gaussian_noise``, {"mean": .0, "stddev": 1.0, "relative": False}).
        exploration_scheduling_options (List[tuple]): A list of 3-tuples specifying the exploration schedulers to be
            registered to the exploration parameters. Each tuple consists of an exploration parameter name, an
            exploration scheduler class (subclass of ``AbsExplorationScheduler``) and keyword arguments for that class.
            The exploration parameter name must be a key in the keyword arguments (second element) of
            ``exploration_strategy``. Defaults to an empty list.
        replay_memory_capacity (int): Capacity of the replay memory. Defaults to 10000.
        random_overwrite (bool): This specifies overwrite behavior when the replay memory capacity is reached. If True,
            overwrite positions will be selected randomly. Otherwise, overwrites will occur sequentially with
            wrap-around. Defaults to False.
        warmup (int): Specifies the minimum rounds to warmup. Defaults to 50000.
        rollout_batch_size (int): Size of the experience batch to use as roll-out information by calling
            ``get_rollout_info``. Defaults to 1000.
        train_batch_size (int): Batch size for training the Q-net. Defaults to 32.
        device (str): Identifier for the torch device. The ``ac_net`` will be moved to the specified device. If it is
            None, the device will be set to "cpu" if cuda is unavailable and "cuda" otherwise. Defaults to None.
    """
    def __init__(
        self,
        name: str,
        ac_net: ContinuousQActorCriticNet,
        reward_discount: float,
        num_epochs: int = 1,
        update_target_every: int = 5,
        q_value_loss_cls: Callable = None,
        q_value_loss_coef: float = 1.0,
        soft_update_coef: float = 1.0,
        exploration_strategy: Tuple[Callable, dict] = (gaussian_noise, {"mean": .0, "stddev": 1.0, "relative": False}),
        exploration_scheduling_options: List[tuple] = None,
        replay_memory_capacity: int = 10000,
        random_overwrite: bool = False,
        warmup: int = 50000,
        rollout_batch_size: int = 1000,
        train_batch_size: int = 32,
        device: str = None
    ) -> None:
        if not isinstance(ac_net, ContinuousQActorCriticNet):
            raise TypeError("model must be an instance of 'ContinuousQActorCriticNet'")
        if exploration_scheduling_options is None:
            exploration_scheduling_options = []

        super(DDPG, self).__init__(name=name, device=device)
        self._ac_net = ac_net.to(self._device)
        self._target_ac_net: ContinuousQActorCriticNet = clone(self._ac_net)
        self._target_ac_net.eval()

        self._reward_discount = reward_discount
        self._num_epochs = num_epochs
        self._update_target_every = update_target_every
        self._q_value_loss_func = q_value_loss_cls() if q_value_loss_cls is not None else torch.nn.MSELoss()
        self._q_value_loss_coef = q_value_loss_coef
        self._soft_update_coef = soft_update_coef
        self._warmup = warmup
        self._rollout_batch_size = rollout_batch_size
        self._train_batch_size = train_batch_size

        self._ac_net_version = 0
        self._target_ac_net_version = 0

        self._exploration_func = exploration_strategy[0]
        self._exploration_params = clone(exploration_strategy[1])
        self._exploration_schedulers = [
            opt[1](self._exploration_params, opt[0], **opt[2]) for opt in exploration_scheduling_options
        ]

        self._replay_memory = ReplayMemory(
            replay_memory_capacity, self._ac_net.state_dim,
            action_dim=self._ac_net.action_dim, random_overwrite=random_overwrite
        )

    def _get_q_values(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self._ac_net.q_critic(
            states=self.ndarray_to_tensor(states),
            actions=self.ndarray_to_tensor(actions)
        ).numpy()

    def _get_state_dim(self) -> int:
        return self._ac_net.state_dim

    def _get_action_range(self) -> Tuple[float, float]:
        return self._ac_net.action_range

    def _call_impl(self, states: np.ndarray) -> np.ndarray:
        if self._replay_memory.size < self._warmup:
            lower, upper = self._ac_net.action_range
            return np.random.uniform(
                low=lower, high=upper,
                size=(states.shape[0], self._ac_net.action_dim)
            )
        else:
            self._ac_net.eval()
            states = self.ndarray_to_tensor(states)
            with torch.no_grad():
                actions = self._ac_net.get_actions(states, exploring=False).cpu().numpy()

            if self._exploring:
                actions = self._exploration_func(states, actions, **self._exploration_params)
            return actions

    def _call_post_check(self, states: np.ndarray, ret: np.ndarray) -> bool:
        return self._shape_check(states, ret)

    def _get_actions_impl(self, states: np.ndarray) -> np.ndarray:
        return self._call_impl(states)

    def _get_action_dim(self) -> int:
        return self._ac_net.action_dim

    def record(
        self,
        agent_id: str,
        state: np.ndarray,
        action: Union[int, float, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        terminal: bool
    ) -> None:
        if next_state is None:
            next_state = np.zeros_like(state, dtype=np.float32)

        self._replay_memory.put(
            np.expand_dims(state, axis=0),
            np.expand_dims(action, axis=0),
            np.expand_dims(reward, axis=0),
            np.expand_dims(next_state, axis=0),
            np.expand_dims(terminal, axis=0)
        )

    def get_rollout_info(self) -> object:
        return self._replay_memory.sample(self._rollout_batch_size)

    def get_batch_loss(self, batch: dict, explicit_grad: bool = False) -> dict:
        self._ac_net.train()
        states: torch.Tensor = self.ndarray_to_tensor(batch["states"])
        next_states: torch.Tensor = self.ndarray_to_tensor(batch["next_states"])
        actual_actions: torch.Tensor = self.ndarray_to_tensor(batch["actions"])
        rewards: torch.Tensor = self.ndarray_to_tensor(batch["rewards"])
        terminals: torch.Tensor = self.ndarray_to_tensor(batch["terminals"]).float()
        if len(actual_actions.shape) == 1:  # TODO: necessary?
            actual_actions = actual_actions.unsqueeze(dim=1)  # [batch_size, 1]

        with torch.no_grad():
            next_q_values = self._target_ac_net.value(next_states)
        target_q_values = (rewards + self._reward_discount * (1 - terminals) * next_q_values).detach()  # [batch_size]

        # loss info
        q_values = self._ac_net.q_critic(states, actual_actions)  # [batch_size]
        q_loss = self._q_value_loss_func(q_values, target_q_values)
        policy_loss = -self._ac_net.value(states).mean()
        loss = policy_loss + self._q_value_loss_coef * q_loss
        loss_info = {
            "policy_loss": policy_loss.detach().cpu().numpy(),
            "q_loss": q_loss.detach().cpu().numpy(),
            "loss": loss.detach().cpu().numpy() if explicit_grad else loss
        }
        if explicit_grad:
            loss_info["grad"] = self._ac_net.get_gradients(loss)

        return loss_info

    def learn_with_data_parallel(self, batch: dict) -> None:
        assert self._task_queue_client, "learn_with_data_parallel is invalid before data_parallel is called."

        self._replay_memory.put(
            batch["states"], batch["actions"], batch["rewards"], batch["next_states"], batch["terminals"]
        )
        for _ in range(self.num_epochs):
            worker_id_list = self._task_queue_client.request_workers()
            batch_list = [
                self._replay_memory.sample(self.train_batch_size // len(worker_id_list))
                for i in range(len(worker_id_list))]
            loss_info_by_policy = self._task_queue_client.submit(
                worker_id_list, batch_list, self.get_state(), self._name)
            # build dummy computation graph by `get_batch_loss` before apply gradients.
            # batch_size=2 because torch.nn.functional.batch_norm doesn't support batch_size=1.
            _ = self.get_batch_loss(self._replay_memory.sample(2), explicit_grad=True)
            self.update(loss_info_by_policy[self._name])

    def update(self, loss_info_list: List[dict]) -> None:
        self._ac_net.apply_gradients(average_grads([loss_info["grad"] for loss_info in loss_info_list]))
        if self._ac_net_version - self._target_ac_net_version == self._update_target_every:
            self._update_target()

    def _update_target(self):
        # soft-update target network
        self._target_ac_net.soft_update(self._ac_net, self._soft_update_coef)
        self._target_ac_net_version = self._ac_net_version

    def learn(self, batch: dict) -> None:
        self._replay_memory.put(
            batch["states"], batch["actions"], batch["rewards"], batch["next_states"], batch["terminals"]
        )
        self.improve()

    def improve(self) -> None:
        for _ in range(self._num_epochs):
            train_batch = self._replay_memory.sample(self._train_batch_size)
            self._ac_net.step(self.get_batch_loss(train_batch)["loss"])
            self._ac_net_version += 1
            if self._ac_net_version - self._target_ac_net_version == self._update_target_every:
                self._update_target()

    def get_state(self) -> object:
        return self._ac_net.get_state()

    def set_state(self, policy_state: object) -> None:
        self._ac_net.set_state(policy_state)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path)
        self._ac_net.set_state(checkpoint["ac_net"])
        self._ac_net_version = checkpoint["ac_net_version"]
        self._target_ac_net.set_state(checkpoint["target_ac_net"])
        self._target_ac_net_version = checkpoint["target_ac_net_version"]
        self._replay_memory = checkpoint["replay_memory"]

    def save(self, path: str) -> None:
        policy_state = {
            "ac_net": self._ac_net.get_state(),
            "ac_net_version": self._ac_net_version,
            "target_ac_net": self._target_ac_net.get_state(),
            "target_ac_net_version": self._target_ac_net_version,
            "replay_memory": self._replay_memory
        }
        torch.save(policy_state, path)
