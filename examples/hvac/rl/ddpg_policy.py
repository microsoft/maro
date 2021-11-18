# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn

from maro.rl.exploration import MultiLinearExplorationScheduler, gaussian_noise
from maro.rl.modeling import ContinuousACNet, ContinuousActionSpace
from maro.rl.policy import DDPG

from .config import config


class AhuACNet(ContinuousACNet):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        actor_lr: float,
        critic_lr: float
    ):
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim

        self.actor = nn.Sequential(
            nn.Linear(self._input_dim, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, self._output_dim),
            nn.Tanh(),
        )
        self.actor_optimizer = torch.optim.Adam(
            params=self.actor.parameters(),
            lr=actor_lr
        )

        self.critic = nn.Sequential(
            nn.Linear(self._input_dim + self._output_dim, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        self.critic_optimizer = torch.optim.Adam(
            params=self.critic.parameters(),
            lr=critic_lr
        )

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def action_dim(self):
        return self._output_dim

    def forward(self, states: torch.tensor, actions: torch.tensor=None) -> torch.tensor:
        if actions is None:
            return self.actor(states)
        else:
            return self.critic(torch.cat([states, actions], dim=1))

    def step(self, loss: torch.tensor):
        raise NotImplementedError

    def set_state(self, state):
        self.actor.load_state_dict(state["actor"]),
        self.critic.load_state_dict(state["critic"]),
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])

    def get_state(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict()
        }


policy_func_dict = {
    "ddpg": lambda name: DDPG(
        name=name,
        ac_net=AhuACNet(
            input_dim=config.state_config["state_dim"],
            output_dim=config.action_config["action_dim"],
            actor_lr=config.hyperparameters["Actor"]["learning_rate"],
            critic_lr=config.hyperparameters["Critic"]["learning_rate"],
        ),
        action_space=ContinuousActionSpace(config.action_config["action_dim"], -1, 1),
        reward_discount=config.hyperparameters["ddpg"]["reward_discount"],
        warmup=config.hyperparameters["ddpg"]["warmup"],
        num_training_epochs=config.hyperparameters["ddpg"]["num_training_epochs"],
        update_target_every=config.hyperparameters["ddpg"]["update_target_every"],
        soft_update_coeff=config.hyperparameters["ddpg"]["soft_update_coeff"],
        replay_memory_capacity=config.replay_memory_config["capacity"],
        random_overwrite=config.replay_memory_config["random_overwrite"],
        train_batch_size=config.replay_memory_config["batch_size"],
        device=config.device,
        exploration_strategy=(
            gaussian_noise,
            {"min_action": -1, "max_action": 1, "stddev": 0}
        ),
        # exploration_scheduling_options=[
        #     (
        #         "stddev",
        #         MultiLinearExplorationScheduler,
        #         {
        #             "start_ep": 0,
        #             "initial_value": 0.1,
        #             "splits": [
        #                 (int(config.num_episode * 0.4), 0.1),
        #                 (int(config.num_episode * 0.8), 0)
        #             ],
        #             "last_ep": config.num_episode - 1,
        #             "final_value": 0
        #         }
        #     ),
        # ],
    )
}
