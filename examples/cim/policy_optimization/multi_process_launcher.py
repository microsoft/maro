# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script is used to debug distributed algorithm in single host multi-process mode.
"""

import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("group_name", help="group name")
    parser.add_argument("num_actors", type=int, help="number of actors")
    args = parser.parse_args()

    learner_path = f"{os.path.split(os.path.realpath(__file__))[0]}/dist_learner.py &"
    actor_path = f"{os.path.split(os.path.realpath(__file__))[0]}/dist_actor.py &"

    # Launch the learner process
    os.system(f"GROUP={args.group_name} NUM_ACTORS={args.num_actors} python " + learner_path)

    # Launch the actor processes
    for _ in range(args.num_actors):
        os.system(f"GROUP={args.group_name} python " + actor_path)
