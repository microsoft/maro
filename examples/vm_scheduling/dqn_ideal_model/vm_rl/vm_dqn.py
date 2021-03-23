import numpy as np
from typing import Union

import torch

from maro.rl.agent import DQN, DQNConfig
from maro.rl.model import SimpleMultiHeadModel
from maro.rl.storage import SimpleStore
from maro.rl.agent import AbsAgent
from maro.utils.exception.rl_toolkit_exception import UnrecognizedTask

class VMDQN(AbsAgent):
    def __init__(
        self,
        name: str,
        model: SimpleMultiHeadModel,
        config: DQNConfig
    ):
        if (config.advantage_mode is not None and
                (model.task_names is None or set(model.task_names) != {"state_value", "advantage"})):
            raise UnrecognizedTask(
                f"Expected model task names 'state_value' and 'advantage' since dueling DQN is used, "
                f"got {model.task_names}"
            )
        super().__init__(
            name, model, config,
            experience_pool=SimpleStore(["state", "action", "reward", "next_state", "next_legal_action", "loss"])
        )
        self._training_counter = 0
        self._target_model = model.copy() if model.is_trainable else None

    def choose_action(self, state: np.ndarray, legal_action: np.ndarray) -> Union[int, np.ndarray]:
        state = torch.from_numpy(state).to(self._device)
        legal_action = torch.from_numpy(legal_action).to(self._device)
        is_single = len(state.shape) == 1
        if is_single:
            state = state.unsqueeze(dim=0)
            legal_action = legal_action.unsqueeze(dim=0)

        q_values = self._get_q_values(self._model, state, is_training=False)
        num_actions = q_values.shape[1]

        greedy_action = self.get_greedy_action(q_values, num_actions, legal_action)

        # No exploration
        if self._config.epsilon == .0:
            return greedy_action[0] if is_single else greedy_action

        if is_single:
            return greedy_action[0] if np.random.random() > self._config.epsilon else self.get_random_action(legal_action[0])

        # batch inference
        return np.array([
            act if np.random.random() > self._config.epsilon else self.get_random_action(legal_action[batch_idx])
            for (batch_idx, act) in enumerate(greedy_action)
        ])

    def get_greedy_action(self, q_values, num_actions, legal_action):
        greedy_action, max_q_value = np.zeros(q_values.shape[0], dtype=np.int), None
        for batch_idx in range(q_values.shape[0]):
            for action_idx in range(num_actions):
                if legal_action[batch_idx, action_idx] == 1:
                    q_value = q_values[batch_idx, action_idx]
                    if max_q_value is None or max_q_value < q_value:
                        max_q_value = q_value
                        greedy_action[batch_idx] = action_idx
        return greedy_action

    def get_random_action(self, legal_action):
        return np.random.choice(np.where(legal_action.cpu().numpy() == 1)[0])

    def _get_q_values(self, model, states, is_training: bool = True):
        if self._config.advantage_mode is not None:
            output = model(states, is_training=is_training)
            state_values = output["state_value"]
            advantages = output["advantage"]
            # Use mean or max correction to address the identifiability issue
            corrections = advantages.mean(1) if self._config.advantage_mode == "mean" else advantages.max(1)[0]
            q_values = state_values + advantages - corrections.unsqueeze(1)
            return q_values
        else:
            return model(states, is_training=is_training)

    def _get_next_q_values(self, current_q_values_for_all_actions, next_states, next_legal_action):
        next_q_values_for_all_actions = self._get_q_values(self._target_model, next_states, is_training=False)
        if self._config.is_double:
            next_illegal_action = (next_legal_action - 1) * 10000000
            actions = (current_q_values_for_all_actions + next_illegal_action).max(dim=1)[1].unsqueeze(1)
            return next_q_values_for_all_actions.gather(1, actions).squeeze(1)  # (N,)
        else:
            return (next_q_values_for_all_actions + next_illegal_action).max(dim=1)[0]   # (N,)

    def _compute_td_errors(self, states, actions, rewards, next_states, next_legal_action):
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(1)  # (N, 1)
        current_q_values_for_all_actions = self._get_q_values(self._model, states)
        current_q_values = current_q_values_for_all_actions.gather(1, actions).squeeze(1)  # (N,)
        next_q_values = self._get_next_q_values(current_q_values_for_all_actions, next_states, next_legal_action)  # (N,)
        target_q_values = (rewards + self._config.reward_discount * next_q_values).detach()  # (N,)
        return self._config.loss_func(current_q_values, target_q_values)

    def train(self):
        if len(self._experience_pool) <= self._config.min_exp_to_train:
            return

        losses = []
        for _ in range(self._config.num_batches):
            indexes, sample = self._experience_pool.sample_by_key("loss", self._config.batch_size)
            state = np.asarray(sample["state"])
            action = np.asarray(sample["action"])
            reward = np.asarray(sample["reward"])
            next_state = np.asarray(sample["next_state"])
            next_legal_action = np.asarray(sample["next_legal_action"])
            loss = self._train_on_batch(state, action, reward, next_state, next_legal_action)
            losses.append(np.mean(loss))
            self._experience_pool.update(indexes, {"loss": list(loss)})
        return losses

    def set_exploration_params(self, epsilon):
        self._config.epsilon = epsilon

    def store_experiences(self, experiences):
        """Store new experiences in the experience pool."""
        self._experience_pool.put(experiences)

    def dump_experience_pool(self, dir_path: str):
        """Dump the experience pool to disk."""
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, self._name), "wb") as fp:
            pickle.dump(self._experience_pool, fp)

    def _train_on_batch(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray, next_legal_action: np.ndarray):
        states = torch.from_numpy(states).to(self._device)
        actions = torch.from_numpy(actions).to(self._device)
        rewards = torch.from_numpy(rewards).to(self._device)
        next_states = torch.from_numpy(next_states).to(self._device)
        next_legal_action = torch.from_numpy(next_legal_action).to(self._device)
        loss = self._compute_td_errors(states, actions, rewards, next_states, next_legal_action)
        self._model.learn(loss.mean().double() if self._config.per_sample_td_error else loss)
        self._training_counter += 1
        if self._training_counter % self._config.target_update_freq == 0:
            self._target_model.soft_update(self._model, self._config.tau)

        return loss.detach().cpu().numpy()
