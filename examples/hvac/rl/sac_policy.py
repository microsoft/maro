import torch
from itertools import chain
from nn_builder.pytorch.NN import NN
from torch.distributions import Normal

from maro.rl.modeling import ContinuousSACNet, ContinuousActionSpace
from maro.rl.policy import SoftActorCritic

from .config import config


def create_NN(input_dim: int, output_dim: int, hyperparameters: dict, seed: int, device: str):
    """Creates a neural network for the agents to use"""
    return NN(
        input_dim=input_dim,
        layers_info=hyperparameters["linear_hidden_units"] + [output_dim],
        output_activation=hyperparameters["output_activation"],
        hidden_activations=hyperparameters["hidden_activations"],
        dropout=hyperparameters["dropout"],
        initialiser=hyperparameters["initialiser"],
        batch_norm=hyperparameters["batch_norm"],
        random_seed=seed
    ).to(device)


class SACNet(ContinuousSACNet):
    def __init__(self, state_dim, action_dim, hyperparameter, device):
        super().__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim

        self.q1 = create_NN(
            input_dim=self.input_dim + self.action_dim,
            output_dim=1,
            hyperparameters=hyperparameter["Critic"],
            seed=hyperparameter["Critic"]["base_seed"],
            device=device
        )
        self.q2 = create_NN(
            input_dim=self.input_dim + self.action_dim,
            output_dim=1,
            hyperparameters=hyperparameter["Critic"],
            seed=hyperparameter["Critic"]["base_seed"] + 1,
            device=device
        )
        self.critic_optimizer = torch.optim.Adam(
            self.q_params,
            lr=hyperparameter["Critic"]["learning_rate"],
            eps=1e-4
        )

        self.actor = create_NN(
            input_dim=self.input_dim,
            output_dim=self.action_dim * 2,
            hyperparameters=hyperparameter["Actor"],
            seed=hyperparameter["Actor"]["seed"],
            device=device
        )
        self.actor_optimizer = torch.optim.Adam(
            self.policy_params,
            lr=hyperparameter["Actor"]["learning_rate"],
            eps=1e-4
        )

        """
        NN(
            (embedding_layers): ModuleList()
            (hidden_layers): ModuleList(
                (0): Linear(in_features=4, out_features=128, bias=True)
                (1): Linear(in_features=128, out_features=128, bias=True)
            )
            (output_layers): ModuleList(
                (0): Linear(in_features=128, out_features=4, bias=True)
            )
            (dropout_layer): Dropout(p=0.3, inplace=False)
        )
        NN(
            (embedding_layers): ModuleList()
            (hidden_layers): ModuleList(
                (0): Linear(in_features=6, out_features=128, bias=True)
                (1): Linear(in_features=128, out_features=128, bias=True)
            )
            (output_layers): ModuleList(
                (0): Linear(in_features=128, out_features=1, bias=True)
            )
            (dropout_layer): Dropout(p=0.3, inplace=False)
        )
        """

    @property
    def input_dim(self):
        return self._state_dim

    @property
    def action_dim(self):
        return self._action_dim

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
        sac_net=SACNet(
            config.state_config["state_dim"],
            config.action_config["action_dim"],
            config.hyperparameters,
            config.device,
        ),
        action_space=ContinuousActionSpace(config.action_config["action_dim"], -1, 1),
        reward_discount=config.hyperparameters["sac"]["reward_discount"],
        alpha=config.hyperparameters["sac"]["alpha"],
        num_training_epochs=config.hyperparameters["sac"]["num_training_epochs"],
        update_target_every=config.hyperparameters["sac"]["update_target_every"],
        soft_update_coeff=config.hyperparameters["sac"]["soft_update_coeff"],
        replay_memory_capacity=config.replay_memory_config["capacity"],
        random_overwrite=config.replay_memory_config["random_overwrite"],
        train_batch_size=config.replay_memory_config["batch_size"],
        warmup=config.hyperparameters["sac"]["warmup"],
        device=config.device
    )
}
