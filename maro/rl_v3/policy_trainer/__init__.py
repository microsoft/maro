from .abs_trainer import AbsTrainer, MultiTrainer, SingleTrainer
from .ac import DiscreteActorCritic
from .ddpg import DDPG
from .discrete_maddpg import DiscreteMADDPG
from .distributed_discrete_maddpg import DistributedDiscreteMADDPG
from .dqn import DQN

__all__ = [
    "AbsTrainer", "MultiTrainer", "SingleTrainer",
    "DiscreteActorCritic",
    "DDPG",
    "DiscreteMADDPG",
    "DistributedDiscreteMADDPG",
    "DQN"
]
