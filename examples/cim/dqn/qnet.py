import numpy as np
import torch
import torch.nn as nn
from maro.rl import DiscreteQNet, OptimOption

class QNet(DiscreteQNet):
    def __init__(self, component: nn.Module, optim_option: OptimOption=None, device='cpu'):
        super().__init__(component, optim_option=optim_option, device=device)

    def forward(self, states):
        states = torch.from_numpy(np.asarray(states)).to(self.device)
        if len(states.shape) == 1:
            states = states.unsqueeze(dim=0)
        return self.component(states)
