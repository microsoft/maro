from typing import Dict

import torch

from maro.rl_v3.model import DiscreteQNet, FullyConnected
from maro.rl_v3.policy import ValueBasedPolicy
from .config import q_net_conf, q_net_optim_conf


class MyQNet(DiscreteQNet):
    def __init__(self) -> None:
        super(MyQNet, self).__init__(state_dim=q_net_conf["input_dim"], action_num=q_net_conf["output_dim"])
        self._fc = FullyConnected(**q_net_conf)
        self._optim = q_net_optim_conf[0](self._fc.parameters(), **q_net_optim_conf[1])

    def _get_q_values_for_all_actions(self, states: torch.Tensor) -> torch.Tensor:
        return self._fc(states)

    def step(self, loss: torch.Tensor) -> None:
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._optim.zero_grad()
        loss.backward()
        return {name: param.grad for name, param in self.named_parameters()}

    def get_net_state(self) -> object:
        return {"network": self.state_dict(), "optim": self._optim.state_dict()}

    def set_net_state(self, net_state: object) -> None:
        assert isinstance(net_state, dict)
        self.load_state_dict(net_state["network"])
        self._optim.load_state_dict(net_state["optim"])


algorithm = "dqn"
get_policy_func_dict = {
    f"{algorithm}.{i}": lambda name: ValueBasedPolicy(name=name, q_net=MyQNet()) for i in range(4)
}
