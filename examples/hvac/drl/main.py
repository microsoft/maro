
import argparse

import numpy as np

from examples.hvac.drl.agents import DDPG, SAC, Trainer
from examples.hvac.drl.hvac_env import MAROHAVEnv
from examples.hvac.drl.config import Config
from examples.hvac.rl.callbacks import post_evaluate

config = Config()


def evaluate(trainer):
    agent_list = trainer.load_model_for_agents()
    assert len(agent_list) == 1
    agent = agent_list[0]

    tracker = {"reward": []}
    done = False
    state = agent.environment.reset()
    while not done:
        if config.algorithm == "ddpg":
            agent.action = agent.pick_action(state=state)
        elif config.algorithm == "sac":
            agent.action = agent.pick_action(eval_ep=True, state=state)
        agent.conduct_action(agent.action)
        state = agent.next_state
        done = agent.done
        tracker["reward"].append(agent.reward)

    for att in ["kw", "at", "dat", "mat"] + ["sps", "das"]:
        tracker[att] = agent.environment.env.snapshot_list["ahus"][::att][1:]
    tracker["total_kw"] = np.cumsum(tracker["kw"])
    tracker["total_reward"] = np.cumsum(tracker["reward"])

    post_evaluate(tracker, -1, config.log_dir, f"drl_{config.algorithm}_eval")

def train(trainer):
    trainer.run_games_for_agents()
    evaluate(trainer)

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--eval', action='store_true', default=False)
    args = argParser.parse_args()

    if config.algorithm == "ddpg":
        agents = [DDPG]
    elif config.algorithm == "sac":
        agents = [SAC]
    else:
        print(f"Wrong algorithm name: {config.algorithm}!")
        exit(0)

    trainer = Trainer(config, agents, MAROHAVEnv())

    if not args.eval:
        train(trainer)
    else:
        evaluate(trainer)
