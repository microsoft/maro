# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script is used to debug distributed algorithm in single-host multi-process mode.
"""

import argparse
import os



if __name__ == "__main__":
    from examples.cim.dqn.components.config import distributed_config
    learner_path = f"{os.path.split(os.path.realpath(__file__))[0]}/learner_launcher.py &"
    actor_path = f"{os.path.split(os.path.realpath(__file__))[0]}/actor_launcher.py &"

    # Launch the actor processes
    for _ in range(distributed_config["peers"]["learner"]["actor"]):
        os.system(f"python {actor_path}")

    # Launch the learner process
    os.system(f"python {learner_path}")
