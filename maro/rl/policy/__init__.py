# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Union

from .abs_policy import AbsPolicy, DummyPolicy, RLPolicy, RuleBasedPolicy
from .continuous_rl_policy import ContinuousRLPolicy
from .discrete_rl_policy import DiscretePolicyGradient, DiscreteRLPolicy, ValueBasedPolicy

GradientPolicy = Union[DiscretePolicyGradient, ContinuousRLPolicy]

__all__ = [
    "AbsPolicy", "DummyPolicy", "RLPolicy", "RuleBasedPolicy",
    "ContinuousRLPolicy",
    "DiscretePolicyGradient", "DiscreteRLPolicy", "ValueBasedPolicy",
    "GradientPolicy",
]
