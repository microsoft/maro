# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from enum import Enum
from typing import Dict


class AllocationMode(Enum):
    BY_POLICY = "by-policy"
    BY_AGENT = "by-agent"
    BY_EXPERIENCE = "by-experience"


class TrainerAllocator(object):
    """Allocate trainers following some strategy."""
    def __init__(self, mode: str, num_trainers: int, policy_names: list, agent2policy: dict):
        assert num_trainers > 0, f"Invalid arguments: num_trainers should be greater than 0 instead of {num_trainers}."
        assert len(policy_names) > 0, "Invalid arguments: policy_names should not be empty."
        assert len(agent2policy) > 0, "Invalid arguments: agent2policy should not be empty."
        self.mode = mode
        self.num_trainers = num_trainers
        self.policy_names = policy_names
        self.agent2policy = agent2policy
        self.trainer_prefix = "GRAD_WORKER"

    def allocate(self, **kwargs):
        logger = kwargs.get('logger', None)
        if self.mode == AllocationMode.BY_POLICY.value:
            policy_names = kwargs.get('policy_names', None)
            return self.allocate_by_policy(policy_names=policy_names, logger=logger)
        elif self.mode == AllocationMode.BY_AGENT.value:
            return self.allocate_by_agent(logger=logger)
        elif self.mode == AllocationMode.BY_EXPERIENCE.value:
            assert 'num_experiences_by_policy' in kwargs
            num_experiences_by_policy = kwargs.pop('num_experiences_by_policy')
            return self.allocate_by_experience(num_experiences_by_policy, logger=logger)
        else:
            raise NotImplementedError(f"{self.mode} is not implemented.")

    def allocate_by_policy(self, policy_names=None, logger=None):
        """Evenly allocate trainers (or grad workers) to each policy."""
        if policy_names is None:
            policy_names = self.policy_names
        num_trainers = self.num_trainers
        policy2trainers = defaultdict(list)
        trainer2policies = defaultdict(list)

        if len(policy_names) >= num_trainers:
            for i, name in enumerate(policy_names):
                trainer_id = i % num_trainers
                policy2trainers[name].append(f"{self.trainer_prefix}.{trainer_id}")
                trainer2policies[f"{self.trainer_prefix}.{trainer_id}"].append(name)
        else:
            trainer_id_list = list(range(num_trainers))
            for i, name in enumerate(policy_names):
                for trainer_id in trainer_id_list[i::len(policy_names)]:
                    policy2trainers[name].append(f"{self.trainer_prefix}.{trainer_id}")
                    trainer2policies[f"{self.trainer_prefix}.{trainer_id}"].append(name)
        return policy2trainers, trainer2policies

    def allocate_by_agent(self, agent2policy=None, logger=None):
        if agent2policy is None:
            agent2policy = self.agent2policy

        num_agents_by_policy = {}
        for agent_id, policy_name in agent2policy.items():
            num_agents_by_policy[policy_name] = num_agents_by_policy.get(policy_name, 0) + 1
        return self._allocate_by_payload(num_agents_by_policy, logger)

    def allocate_by_experience(self, num_experiences_by_policy: dict, logger=None):
        return self._allocate_by_payload(num_experiences_by_policy, logger)

    def _allocate_by_payload(self, num_payload: Dict[str, int], logger=None):
        """Allocate trainers (or grad workers) by payload of each policy.

        Args:
            num_payload (Dict[str, int]): Payload of each policy, could be experience numbers
                or agent nums.

        Returns:
            policy2trainers (Dict[str, list]): The mapping from policy name to assigned trainer ids.
            trainer2policies (Dict[str, list]): The mapping from trainer id to according policies.
        """
        num_trainers = self.num_trainers
        policy2trainers = defaultdict(list)
        trainer2policies = defaultdict(list)

        # no payload yet
        if len(num_payload) == 0:
            return self.allocate_by_policy()
        # allocate trainers according to historical payload.
        else:
            total_num_payload = sum(num_payload.values())
            average_payload = total_num_payload / num_trainers

            offset = 0
            policy_quota = dict()
            for name, payload in num_payload.items():
                quota = payload / average_payload
                quota = max(1, int(round(quota)))
                policy_quota[name] = quota

            # adjust quota if any redundancy occurs.
            redundancy = num_trainers - sum(policy_quota.values())
            if redundancy > 0:
                busiest_policy = max(policy_quota, key=lambda name: policy_quota[name])
                policy_quota[busiest_policy] += redundancy

            for name, quota in policy_quota.items():
                if logger is not None:
                    logger.info(
                        f"policy {name} payload: {num_payload[name]},  quota: {quota} node(s)")
                for i in range(quota):
                    trainer_id = (i + offset) % num_trainers
                    policy2trainers[name].append(f"{self.trainer_prefix}.{trainer_id}")
                    trainer2policies[f"{self.trainer_prefix}.{trainer_id}"].append(name)
                offset = (offset + quota) % num_trainers

        return policy2trainers, trainer2policies
