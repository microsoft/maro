# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from enum import Enum
from typing import Dict


class AllocationMode(Enum):
    BY_POLICY = "by-policy"
    BY_AGENT = "by-agent"
    BY_EXPERIENCE = "by-experience"


class WorkerAllocator(object):
    """Allocate workers following some strategy."""
    def __init__(self, mode: str, num_workers: int, policy_names: list, agent2policy: dict):
        assert num_workers > 0, f"Invalid arguments: num_workers should be greater than 0 instead of {num_workers}."
        assert len(policy_names) > 0, "Invalid arguments: policy_names should not be empty."
        assert len(agent2policy) > 0, "Invalid arguments: agent2policy should not be empty."
        self.mode = mode
        self.num_workers = num_workers
        self.policy_names = policy_names
        self.agent2policy = agent2policy
        self.worker_prefix = "GRAD_WORKER"
        self.worker_list = [f"{self.worker_prefix}.{i}" for i in range(self.num_workers)]
        self.logger = None

        self._cached_mappings = dict()

    def set_logger(self, logger):
        self.logger = logger

    def allocate(self, **kwargs):
        if self.mode not in self._cached_mappings:
            if self.mode == AllocationMode.BY_POLICY.value:
                self._cached_mappings[self.mode] = self.allocate_by_policy()
            elif self.mode == AllocationMode.BY_AGENT.value:
                self._cached_mappings[self.mode] = self.allocate_by_agent()
            elif self.mode == AllocationMode.BY_EXPERIENCE.value:
                assert 'num_experiences_by_policy' in kwargs
                num_experiences_by_policy = kwargs.pop('num_experiences_by_policy')
                self._cached_mappings[self.mode] = self.allocate_by_experience(num_experiences_by_policy)
            else:
                raise NotImplementedError(f"{self.mode} is not implemented.")
        return self._cached_mappings[self.mode]

    def allocate_by_policy(self):
        """Evenly allocate grad workers to each policy."""
        policy_names = self.policy_names
        num_workers = self.num_workers
        policy2workers = defaultdict(list)
        worker2policies = defaultdict(list)

        if len(policy_names) >= num_workers:
            for i, name in enumerate(policy_names):
                worker_id = i % num_workers
                policy2workers[name].append(f"{self.worker_prefix}.{worker_id}")
                worker2policies[f"{self.worker_prefix}.{worker_id}"].append(name)
        else:
            worker_id_list = list(range(num_workers))
            for i, name in enumerate(policy_names):
                for worker_id in worker_id_list[i::len(policy_names)]:
                    policy2workers[name].append(f"{self.worker_prefix}.{worker_id}")
                    worker2policies[f"{self.worker_prefix}.{worker_id}"].append(name)
        return policy2workers, worker2policies

    def allocate_by_agent(self):
        agent2policy = self.agent2policy
        num_agents_by_policy = {}
        for agent_id, policy_name in agent2policy.items():
            num_agents_by_policy[policy_name] = num_agents_by_policy.get(policy_name, 0) + 1
        return self._allocate_by_payload(num_agents_by_policy)

    def allocate_by_experience(self, num_experiences_by_policy: dict):
        return self._allocate_by_payload(num_experiences_by_policy)

    def _allocate_by_payload(self, num_payload: Dict[str, int]):
        """Allocate grad workers by payload of each policy.

        Args:
            num_payload (Dict[str, int]): Payload of each policy, could be experience numbers
                or agent nums.

        Returns:
            policy2workers (Dict[str, list]): The mapping from policy name to assigned worker ids.
            worker2policies (Dict[str, list]): The mapping from worker id to according policies.
        """
        num_workers = self.num_workers
        policy2workers = defaultdict(list)
        worker2policies = defaultdict(list)

        # no payload yet
        if len(num_payload) == 0:
            return self.allocate_by_policy()
        # allocate workers according to historical payload.
        else:
            total_num_payload = sum(num_payload.values())
            average_payload = total_num_payload / num_workers

            offset = 0
            policy_quota = dict()
            for name, payload in num_payload.items():
                quota = payload / average_payload
                quota = max(1, int(round(quota)))
                policy_quota[name] = quota

            # adjust quota if any redundancy occurs.
            redundancy = num_workers - sum(policy_quota.values())
            if redundancy > 0:
                busiest_policy = max(policy_quota, key=lambda name: policy_quota[name])
                policy_quota[busiest_policy] += redundancy

            for name, quota in policy_quota.items():
                if self.logger is not None:
                    self.logger.info(
                        f"policy {name} payload: {num_payload[name]},  quota: {quota} node(s)")
                for i in range(quota):
                    worker_id = (i + offset) % num_workers
                    policy2workers[name].append(f"{self.worker_prefix}.{worker_id}")
                    worker2policies[f"{self.worker_prefix}.{worker_id}"].append(name)
                offset = (offset + quota) % num_workers

        return policy2workers, worker2policies
