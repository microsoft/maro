# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod

import torch

from maro.rl.model import AbsCoreModel
from maro.rl.storage import SimpleStore


class AgentConfig:
    """Configuration for the DQN algorithm.

    Args:
        reward_discount (float): Reward decay as defined in standard RL terminology.
        experience_memory_size (int): Size of the experience memory. If it is -1, the experience memory is of
            unlimited size.
        experience_memory_overwrite_type (str): A string indicating how experiences in the experience memory are
            to be overwritten after its capacity has been reached. Must be "rolling" or "random".

    """
    __slots__ = ["reward_discount", "experience_memory_size", "experience_memory_overwrite_type"]

    def __init__(self, reward_discount: float, experience_memory_size: int, experience_memory_overwrite_type: str):
        self.reward_discount = reward_discount
        self.experience_memory_size = experience_memory_size
        self.experience_memory_overwrite_type = experience_memory_overwrite_type


class AbsAgent(ABC):
    """Abstract RL agent class.

    It's a sandbox for the RL algorithm. Scenario-specific details will be excluded.
    We focus on the abstraction algorithm development here. Environment observation and decision events will
    be converted to a uniform format before calling in. The output will be converted to an environment
    executable format before return back to the environment. Its key responsibility is optimizing policy based
    on interaction with the environment.

    Args:
        model (AbsCoreModel): Task model or container of task models required by the algorithm.
        config: Settings for the algorithm.
    """
    def __init__(self, model: AbsCoreModel, config: AgentConfig):
        self.model = model
        self.config = config
        self.experience_memory = SimpleStore(
            ["S", "A", "R", "S_"],
            capacity=self.config.experience_memory_size,
            overwrite_type=self.config.experience_memory_overwrite_type
        )
        self.device = None

    def to_device(self, device):
        self.device = device
        self.model = self.model.to(device)

    @abstractmethod
    def choose_action(self, state):
        """This method uses the underlying model(s) to compute an action from a shaped state.

        Args:
            state: A state object shaped by a ``StateShaper`` to conform to the model input format.

        Returns:
            The action to be taken given ``state``. It is usually necessary to use an ``ActionShaper`` to convert
            this to an environment executable action.
        """
        return NotImplementedError

    def set_exploration_params(self, **params):
        pass

    def store_experiences(self, experiences: dict):
        """Pull experiences from the replay memory stored by an environment wrapper."""
        if set(experiences) != {"S", "A", "R", "S_"}:
            raise ValueError("The keys of experiences must be {'S', 'A', 'R', 'S_'}")
        self.experience_memory.put(experiences)

    @abstractmethod
    def learn(self):
        """Algorithm-specific training logic.

        The parameters are data to train the underlying model on. Algorithm-specific loss and optimization
        should be reflected here.
        """
        return NotImplementedError

    def load_model(self, model):
        """Load models from memory."""
        self.model.load_state_dict(model)

    def dump_model(self):
        """Return the algorithm's trainable models."""
        return self.model.state_dict()

    def load_model_from_file(self, path: str):
        """Load trainable models from disk.

        Load trainable models from the specified directory. The model file is always prefixed with the agent's name.

        Args:
            path (str): path to the directory where the models are saved.
        """
        self.model.load_state_dict(torch.load(path))

    def dump_model_to_file(self, path: str):
        """Dump the algorithm's trainable models to disk.

        Dump trainable models to the specified directory. The model file is always prefixed with the agent's name.

        Args:
            path (str): path to the directory where the models are saved.
        """
        torch.save(self.model.state_dict(), path)
