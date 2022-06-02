# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Any, Dict, List

from maro.rl.policy import AbsPolicy
from maro.rl.rollout import AbsEnvSampler
from maro.rl.training import AbsTrainer


@dataclass
class RLComponentBundle:
    """Bundle of all necessary components to run a RL job in MARO.

    env_sampler (AbsEnvSampler): Environment sampler of the scenario.
    agent2policy (Dict[Any, str]): Agent name to policy name mapping of the RL job. For example:
        {agent1: policy1, agent2: policy1, agent3: policy2}.
    policies (List[AbsPolicy]): Policies.
    trainers (List[AbsTrainer]): Trainers.
    device_mapping (Dict[str, str], default=None): Device mapping that identifying which device to put each policy.
        If None, there will be no explicit device assignment.
    policy_trainer_mapping (Dict[str, str], default=None): Policy-trainer mapping which identifying which trainer to
        train each policy. If None, then a policy's trainer's name is the first segment of the policy's name,
        seperated by dot. For example, "ppo_1.policy" is trained by "ppo_1". Only policies that provided in
        policy-trainer mapping are considered as trainable polices. Policies that not provided in policy-trainer
        mapping will not be trained.
    """

    env_sampler: AbsEnvSampler
    agent2policy: Dict[Any, str]
    policies: List[AbsPolicy]
    trainers: List[AbsTrainer]
    device_mapping: Dict[str, str] = None
    policy_trainer_mapping: Dict[str, str] = None

    def __post_init__(self) -> None:
        # Check missing policies
        policy_set = set([policy.name for policy in self.policies])
        not_found = [policy_name for policy_name in self.agent2policy.values() if policy_name not in policy_set]
        if len(not_found) > 0:
            raise ValueError(f"The following policies are required but cannot be found: [{', '.join(not_found)}]")

        # Remove unused policies
        kept_policies = []
        for policy in self.policies:
            if policy.name not in self.agent2policy.values():
                raise Warning(f"Policy {policy.name} if removed since it is not used by any agent.")
            else:
                kept_policies.append(policy)
        self.policies = kept_policies
        policy_set = set([policy.name for policy in self.policies])

        if self.device_mapping is not None:
            self.device_mapping = {k: v for k, v in self.device_mapping.items() if k in policy_set}
        else:
            self.device_mapping = {}

        # Create default policy-trainer mapping if not provided
        if self.policy_trainer_mapping is None:  # Default policy-trainer naming rule
            self.policy_trainer_mapping = {policy_name: policy_name.split(".")[0] for policy_name in policy_set}

        # Check missing trainers
        self.policy_trainer_mapping = {
            policy_name: trainer_name
            for policy_name, trainer_name in self.policy_trainer_mapping.items()
            if policy_name in policy_set
        }
        trainer_set = set([trainer.name for trainer in self.trainers])
        not_found = [
            trainer_name for trainer_name in self.policy_trainer_mapping.values() if trainer_name not in trainer_set
        ]
        if len(not_found) > 0:
            raise ValueError(f"The following trainers are required but cannot be found: [{', '.join(not_found)}]")

        # Remove unused trainers
        kept_trainers = []
        for trainer in self.trainers:
            if trainer.name not in self.policy_trainer_mapping.values():
                raise Warning(f"Trainer {trainer.name} if removed since no policy is trained by it.")
            else:
                kept_trainers.append(trainer)
        self.trainers = kept_trainers

    @property
    def trainable_agent2policy(self) -> Dict[Any, str]:
        return {
            agent_name: policy_name
            for agent_name, policy_name in self.agent2policy.items()
            if policy_name in self.policy_trainer_mapping
        }

    @property
    def trainable_policies(self) -> List[AbsPolicy]:  # TODO: Abs or RL?
        return [policy for policy in self.policies if policy.name in self.policy_trainer_mapping]
