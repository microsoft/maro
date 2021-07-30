# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import getenv
from os.path import dirname, realpath

from maro.rl.learning import SimpleLearner


workflow_dir = dirname((realpath(__file__)))
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from agent_wrapper import get_agent_wrapper
from general import (
    get_env_wrapper, get_eval_env_wrapper, log_dir, post_collect, post_evaluate, post_update, update_trigger, warmup
)


if __name__ == "__main__":
    num_episodes = getenv("NUMEPISODES")
    if num_episodes is None:
        raise ValueError("Missing envrionment variable: NUMEPISODES")

    num_steps = int(getenv("NUMSTEPS", default=-1))
    SimpleLearner(
        get_env_wrapper(),
        get_agent_wrapper(local_update=True),
        num_episodes,
        num_steps=num_steps,
        eval_env=get_eval_env_wrapper(),
        eval_schedule=int(getenv("EVALSCH")),
        update_trigger=update_trigger,
        warmup=warmup,
        post_collect=post_collect,
        post_evaluate=post_evaluate,
        post_update=post_update,
        log_dir=log_dir
    ).run()
