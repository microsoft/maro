import numpy as np
import torch
import torch.nn as nn
from maro.rl import DQN, DQNConfig, DiscreteQNet, ExperienceManager, FullyConnectedBlock, OptimOption


class QNet(DiscreteQNet):
    def __init__(self, component: nn.Module, optim_option: OptimOption=None, device=None):
        super().__init__(component, optim_option=optim_option, device=device)

    def forward(self, states):
        states = torch.from_numpy(np.asarray(states)).to(self.device)
        if len(states.shape) == 1:
            states = states.unsqueeze(dim=0)
        return self.component(states)


def get_independent_policy(cfg, name, training: bool = True):
    qnet = QNet(
        FullyConnectedBlock(**cfg["model"]["network"]),
        optim_option=OptimOption(**cfg["model"]["optimization"])
    )
    exp_cfg = cfg["experience_manager"]["training"] if training else cfg["experience_manager"]["rollout"]
    return DQN(
        name=name,
        q_net=qnet,
        experience_manager=ExperienceManager(**exp_cfg),
        config=DQNConfig(**cfg["algorithm_config"]),
        update_trigger=cfg["update_trigger"],
        warmup=cfg["warmup"]
    )