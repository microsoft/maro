# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class DistributedTrainingMode(Enum):
    LEARNER_ACTOR = "learner_actor"
    ACTOR_TRAINER = "actor_trainer"
