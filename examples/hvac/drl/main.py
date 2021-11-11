
import argparse
import numpy as np
import os
import pickle
import random

from maro.utils import Logger

from examples.hvac.drl.agents import DDPG, SAC
from examples.hvac.drl.hvac_env import MAROHAVEnv
from examples.hvac.drl.callbacks import visualize_rolling_scores
from examples.hvac.drl.config import Config
from examples.hvac.rl.callbacks import post_evaluate


class Trainer(object):
    def __init__(self, config, agent_class, env):
        if config.randomize_random_seed:
            config.seed = random.randint(0, 2**32 -2)

        self.config = config
        self.env = env
        self.logger = Logger(tag="Trainer", dump_folder=self.config.log_dir)
        self.agent = agent_class(config, env, self.logger)

        self.logger.info(f"RANDOM SEED: {config.seed}")

    def load_model(self):
        self.agent.load_model()

    def train(self):
        self.env.reset()
        game_scores, rolling_scores, time_taken = self.agent.run_n_episodes()

        results = [game_scores, rolling_scores, len(rolling_scores), -1 * max(rolling_scores), time_taken]
        with open(os.path.join(self.config.log_dir, "results.pkl"), "wb") as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

        visualize_rolling_scores(
            rolling_scores,
            self.config.log_dir,
            f"{self.agent.environment_title} - {self.agent.agent_name}"
        )

    def evaluate(self):
        tracker = {"reward": []}
        done = False
        state = self.agent.environment.reset()
        while not done:
            if config.algorithm == "ddpg":
                self.agent.action = self.agent.pick_action(state=state)
            elif config.algorithm == "sac":
                self.agent.action = self.agent.pick_action(eval_ep=True, state=state)
            self.agent.conduct_action(self.agent.action)
            state = self.agent.next_state
            done = self.agent.done
            tracker["reward"].append(self.agent.reward)

        for att in ["kw", "at", "dat", "mat"] + ["sps", "das"]:
            tracker[att] = self.agent.environment.env.snapshot_list["ahus"][::att][1:]
        tracker["total_kw"] = np.cumsum(tracker["kw"])
        tracker["total_reward"] = np.cumsum(tracker["reward"])

        post_evaluate(tracker, -1, config.log_dir, f"drl_{config.algorithm}_eval")


if __name__ == "__main__":
    config = Config()

    argParser = argparse.ArgumentParser()
    argParser.add_argument('--eval', action='store_true', default=False)
    args = argParser.parse_args()

    if config.algorithm == "ddpg":
        agent_class = DDPG
    elif config.algorithm == "sac":
        agent_class = SAC
    else:
        print(f"Wrong algorithm name: {config.algorithm}!")
        exit(0)

    trainer = Trainer(config, agent_class, MAROHAVEnv())

    if not args.eval:
        trainer.train()
        trainer.evaluate()
    else:
        trainer.load_model()
        trainer.evaluate()
