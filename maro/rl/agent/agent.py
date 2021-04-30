# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import Dict, Union

import torch

from maro.rl.model import AbsCoreModel
from maro.rl.storage import SimpleStore

from .experience_enum import Experience


class GenericAgentConfig:
    """Generic agent settings. 

    Args:
        experience_memory_size (int): Size of the experience memory. If it is -1, the experience memory is of
            unlimited size.
        experience_memory_overwrite_type (str): A string indicating how experiences in the experience memory are
            to be overwritten after its capacity has been reached. Must be "rolling" or "random".
        empty_experience_memory_after_step (bool): If True, the experience memory will be emptied after each call
            to ``step``.
        min_new_experiences_to_trigger_learning (int): Minimum number of new experiences required to trigger learning.
            Defaults to 1.
        min_experiences_to_trigger_learning (int): Minimum number of experiences in the experience memory required for
            training. Defaults to 1.
    """
    __slots__ = [
        "experience_memory_size", "experience_memory_overwrite_type", "empty_experience_memory_after_step",
        "min_new_experiences_to_trigger_learning", "min_experiences_to_trigger_learning"
    ]

    def __init__(
        self,
        experience_memory_size: int,
        experience_memory_overwrite_type: str,
        empty_experience_memory_after_step: bool,
        min_new_experiences_to_trigger_learning: int = 1,
        min_experiences_to_trigger_learning: int = 1
    ):
        self.experience_memory_size = experience_memory_size
        self.experience_memory_overwrite_type = experience_memory_overwrite_type
        self.empty_experience_memory_after_step = empty_experience_memory_after_step
        self.min_new_experiences_to_trigger_learning = min_new_experiences_to_trigger_learning
        self.min_experiences_to_trigger_learning = min_experiences_to_trigger_learning


class AbsAgent(ABC):
    """Abstract RL agent class.

    It's a sandbox for the RL algorithm. Scenario-specific details will be excluded.
    We focus on the abstraction algorithm development here. Environment observation and decision events will
    be converted to a uniform format before calling in. The output will be converted to an environment
    executable format before return back to the environment. Its key responsibility is optimizing policy based
    on interaction with the environment.

    Args:
        model (AbsCoreModel): Task model or container of task models required by the algorithm.
        algorithm_config: Algorithm-specific configuration.
        generic_config (GenericAgentConfig): Non-algorithm-specific configuration.  
        experience_memory (SimpleStore): Experience memory for the agent. If None, an experience memory will be
            created at init time. Defaults to None.
    """
    def __init__(
        self,
        model: AbsCoreModel,
        algorithm_config,
        generic_config: GenericAgentConfig,
        experience_memory: SimpleStore = None
    ):
        self.model = model
        self.algorithm_config = algorithm_config
        self.generic_config = generic_config
        if not experience_memory:
            self.experience_memory = SimpleStore(
                [Experience.STATE, Experience.ACTION, Experience.REWARD, Experience.NEXT_STATE],
                capacity=self.generic_config.experience_memory_size,
                overwrite_type=self.generic_config.experience_memory_overwrite_type
            )
        else:
            self.experience_memory = experience_memory

        self.device = torch.device('cpu')

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

    def learn(self, experiences: dict) -> bool:
        """Store experinces in the experience memory and train the model if necessary."""
        expected_keys = set(self.experience_memory.keys)
        if set(experiences.keys()) != set(self.experience_memory.keys):
            raise ValueError(f"The keys of experiences must be {expected_keys}")
        self.experience_memory.put(experiences)
        if (
            len(experiences[Experience.STATE]) >= self.generic_config.min_new_experiences_to_trigger_learning and
            len(self.experience_memory) >= self.generic_config.min_experiences_to_trigger_learning
        ):
            self.step()
            if self.generic_config.empty_experience_memory_after_step:
                self.experience_memory.clear()
            return True
        return False

    @abstractmethod
    def step(self):
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


class AgentGroup:
    """Convenience wrapper of a set of agents that share the same underlying model.

    Args:
        agent_dict (Union[AbsAgent, dict]): A single agent or a homogeneous set of agents that
            share the same underlying model instance.
    """

    def __init__(
        self,
        agent_names: list,
        agent_cls,
        model,
        algorithm_config,
        generic_config: Union[GenericAgentConfig, Dict[str, GenericAgentConfig]],
        experience_memory: Union[SimpleStore, Dict[str, SimpleStore]] = None
    ):
        self._members = agent_names
        self._validate_agent_config(algorithm_config)
        self._validate_agent_config(generic_config)
        self.model = model

        def get_per_agent_obj(obj, agent_name):
            return obj[agent_name] if isinstance(obj, dict) else obj

        self.agent_dict = {
            name: agent_cls(
                self.model,
                get_per_agent_obj(algorithm_config, name),
                get_per_agent_obj(generic_config, name),
                experience_memory=get_per_agent_obj(experience_memory, name)
            )
            for name in agent_names
        }

    def __getitem__(self, agent_id):
        if len(self.agent_dict) == 1:
            return self.agent_dict["AGENT"]
        else:
            return self.agent_dict[agent_id]

    def __len__(self):
        return len(self.agent_dict)

    @property
    def members(self):
        return self._members

    def choose_action(self, state_by_agent: dict):
        return {agent_id: self.agent_dict[agent_id].choose_action(state) for agent_id, state in state_by_agent.items()}

    def set_exploration_params(self, params):
        # Per-agent exploration parameters
        if isinstance(params, dict) and params.keys() <= self.agent_dict.keys():
            for agent_id, params in params.items():
                self.agent_dict[agent_id].set_exploration_params(**params)
        # Shared exploration parameters for all agents
        else:
            for agent in self.agent_dict.values():
                agent.set_exploration_params(**params)

    def learn(self, experiences: dict) -> set:
        """Store experiences in the agents' experience memory.

        The top-level keys of ``experiences`` will be treated as agent IDs.
        """
        return {agent_id for agent_id, exp in experiences.items() if self.agent_dict[agent_id].learn(exp)}

    def step(self):
        for agent_id in agent_ids:
            self.agent_dict[agent_id].step()

    def load_model(self, model):
        """Load models from memory."""
        self.model.load_state_dict(model)
        assert all(agent.model.state_dict() == self.model.state_dict() for agent in self.agent_dict.values())

    def dump_model(self):
        """Get agents' underlying models.

        This is usually used in distributed mode where models need to be broadcast to remote roll-out actors.
        """
        return self.model.state_dict()

    def load_model_from_file(self, path):
        """Load models from disk."""
        self.model.load_state_dict(torch.load(path))
        assert all(agent.model.state_dict() == self.model.state_dict() for agent in self.agent_dict.values())

    def dump_model_to_file(self, path: str):
        """Dump agents' models to disk.

        If the agents don't share models, each agent will use its own name to create a separate file under ``path``
        for dumping.
        """
        torch.save(self.model.state_dict(), path)

    def _validate_agent_config(self, agent_config):
        if isinstance(agent_config, dict):
            expected_agent_names = set(self._members)
            agent_names_in_config = set(agent_config.keys())
            if expected_agent_names != agent_names_in_config:
                raise ValueError(f"Expected {expected_agent_names} as config keys, got {agent_names_in_config}")
