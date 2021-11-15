from itertools import chain
import torch
from torch.distributions import Normal

from maro.rl.modeling import ContinuousSACNet
from maro.rl.policy import SoftActorCritic

from .config import (
    state_dim, action_dim, action_lower_bound, action_upper_bound, sac_config
)
from ..drl.config import Config
from ..drl.agents.model import create_NN

config = Config()


class SACNet(ContinuousSACNet):
    def __init__(self):
        super().__init__()

        self.q1 = create_NN(
            input_dim=self.input_dim + self.action_dim,
            output_dim=1,
            hyperparameters=config.hyperparameters["Critic"],
            seed=config.seed,
            device="cpu"
        )
        self.q2 = create_NN(
            input_dim=self.input_dim + self.action_dim,
            output_dim=1,
            hyperparameters=config.hyperparameters["Critic"],
            seed=config.seed + 1,
            device="cpu"
        )
        self.critic_optimizer = torch.optim.Adam(
            self.q_params,
            lr=config.hyperparameters["Critic"]["learning_rate"],
            eps=1e-4
        )

        self.actor = create_NN(
            input_dim=self.input_dim,
            output_dim=self.action_dim * 2,
            hyperparameters=config.hyperparameters["Actor"],
            seed=config.seed,
            device="cpu"
        )
        self.actor_optimizer = torch.optim.Adam(
            self.policy_params,
            lr=config.hyperparameters["Actor"]["learning_rate"],
            eps=1e-4
        )

        self._action_base = torch.tensor(action_lower_bound)
        self._action_range = torch.tensor(action_upper_bound) - torch.tensor(action_lower_bound)

    @property
    def input_dim(self):
        return state_dim

    @property
    def action_dim(self):
        return action_dim

    @property
    def action_min(self):
        return action_lower_bound

    @property
    def action_max(self):
        return action_upper_bound

    @property
    def policy_params(self):
        return self.actor.parameters()

    @property
    def q_params(self):
        return chain(self.q1.parameters(), self.q2.parameters())

    @property
    def policy_optim(self):
        return self.actor_optimizer

    @property
    def q_optim(self):
        return self.critic_optimizer

    def __call__(self, states: torch.tensor, deterministic: bool) -> torch.tensor:
        return self.forward(states, deterministic)

    def forward(self, states: torch.tensor, deterministic: bool) -> torch.tensor:
        actor_output = self.actor(states)

        mean = torch.clamp(actor_output[:, :self.action_dim], -3.0, 3.0)
        log_std = torch.clamp(actor_output[:, self.action_dim:], -5.0, 2.0)
        std = log_std.exp()
        distribution = Normal(mean, std)

        x = distribution.rsample()
        action = torch.tanh(x)
        log_prob = distribution.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1)

        if deterministic:
            action = torch.tanh(mean)

        action = self._action_base + self._action_range * (action + 1) / 2

        return action, log_prob

    def get_action(self, states: torch.tensor, deterministic: bool = False) -> torch.tensor:
        """Compute actions given a batch of states."""
        return self.forward(states, deterministic=deterministic)[0]

    def get_q1_values(self, states: torch.tensor, actions: torch.tensor):
        input = torch.cat([states, actions], dim=-1)
        return self.q1(input).squeeze(dim=-1)

    def get_q2_values(self, states: torch.tensor, actions: torch.tensor):
        input = torch.cat([states, actions], dim=-1)
        return self.q2(input).squeeze(dim=-1)

    def step(self, loss: torch.tensor):
        raise NotImplementedError

    def get_state(self):
        return {
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "actor": self.actor.state_dict(),
            "critic_optim": self.critic_optimizer.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict()
        }

    def set_state(self, state):
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self.actor.load_state_dict(state["actor"])
        self.critic_optimizer.load_state_dict(state["critic_optim"])
        self.actor_optimizer.load_state_dict(state["actor_optim"])


policy_func_dict = {
    "sac": lambda name: SoftActorCritic(
        name=name,
        sac_net=SACNet(),
        **sac_config
    )
}
