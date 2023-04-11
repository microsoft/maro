from typing import Any, Callable, Dict, List, Optional

import torch

from maro.rl_v31.policy.base import AbsPolicy
from maro.rl_v31.rollout.wrapper import EnvWrapper
from maro.rl_v31.training.trainer import BaseTrainer


class RLComponentBundle(object):
    def __init__(
        self,
        env_wrapper_func: Callable[[], EnvWrapper],
        agent2policy: Dict[Any, str],
        policies: List[AbsPolicy],
        trainers: List[BaseTrainer],
        policy2trainer: Optional[Dict[str, str]] = None,
        policy_device_mapping: Optional[Dict[str, str]] = None,
        trainer_device_mapping: Optional[Dict[str, str]] = None,
        metrics_agg_func: Optional[Callable[[List[dict]], dict]] = None,
    ) -> None:
        self.env_wrapper_func = env_wrapper_func
        self.agent2policy = agent2policy

        #
        required_policies = set(agent2policy.values())
        provided_policies = set(policy.name for policy in policies)
        missing_policies = required_policies - provided_policies
        if len(missing_policies) > 0:
            raise ValueError(f"Policies [{', '.join(sorted(missing_policies))}] are missing.")
        useless_policies = provided_policies - required_policies
        if len(useless_policies) > 0:
            raise Warning(
                f"Policies [{', '.join(sorted(useless_policies))}] are ignored since they are not used by any agent.",
            )
        self.policies = [policy for policy in policies if policy.name in required_policies]
        self.policy_dict: Dict[str, AbsPolicy] = {policy.name: policy for policy in self.policies}

        #
        self.policy2trainer = (
            policy2trainer
            if policy2trainer is not None
            else {policy.name: policy.name.split(".")[0] for policy in self.policies}
        )
        required_trainers = set(self.policy2trainer.values())
        provided_trainers = set(trainer.name for trainer in trainers)
        missing_trainers = required_trainers - provided_trainers
        if len(missing_trainers) > 0:
            raise ValueError(f"Trainers [{', '.join(sorted(missing_trainers))}] are missing.")
        useless_trainers = provided_trainers - required_trainers
        if len(useless_trainers) > 0:
            raise Warning(
                f"Trainers [{', '.join(sorted(useless_trainers))}] are ignored since they are not used by any policy.",
            )
        self.trainers = [trainer for trainer in trainers if trainer.name in required_trainers]
        self.trainer_dict: Dict[str, BaseTrainer] = {trainer.name: trainer for trainer in self.trainers}

        #
        self.policy_device_mapping: Dict[str, torch.device] = (
            {
                policy_name: torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
                for policy_name, device in policy_device_mapping.items()
                if policy_name in self.policy_dict
            }
            if policy_device_mapping is not None
            else {}
        )

        self.trainer_device_mapping: Dict[str, torch.device] = (
            {
                trainer_name: torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
                for trainer_name, device in trainer_device_mapping.items()
                if trainer_name in self.trainer_dict
            }
            if trainer_device_mapping is not None
            else {}
        )

        #
        if metrics_agg_func is None:
            self.metrics_agg_func = lambda x: {"metrics": x}
        else:
            self.metrics_agg_func = metrics_agg_func
