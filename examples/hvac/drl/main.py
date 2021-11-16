
import argparse
import numpy as np
import os

from maro.utils import Logger

from examples.hvac.drl.agents import DDPG, SAC
from examples.hvac.drl.hvac_env import MAROHAVEnv
from examples.hvac.rl.callbacks import post_evaluate
from examples.hvac.rl.config import config
from examples.hvac.rl.callbacks import baseline, visualize_returns

LOG_DIR = os.path.join(config.training_config["log_path"], config.experiment_name)

class Trainer(object):
    def __init__(self, config, agent_class, env):
        self.config = config
        self.env = env
        self.logger = Logger(tag="Trainer", dump_folder=LOG_DIR)
        self.agent = agent_class(config, env, self.logger)

    def load_model(self):
        self.agent.load_model()

    def train(self):
        self.env.reset()
        returns, rolling_returns, _ = self.agent.run_n_episodes()

        visualize_returns(
            rolling_returns,
            LOG_DIR,
            f"Rolling Returns - {self.agent.agent_name}"
        )

        visualize_returns(
            returns,
            LOG_DIR,
            f"Returns - {self.agent.agent_name}"
        )

    def evaluate(self):
        self.agent.reset_game()
        tracker = {"reward": []}
        done = False
        state = self.agent.state
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

        res = post_evaluate(tracker, -1, LOG_DIR, f"drl_{config.algorithm}_eval")
        self.logger.info(f"Final improvement: {res}")


if __name__ == "__main__":
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

    trainer = Trainer(config, agent_class, MAROHAVEnv(config.env_config, baseline))

    if not args.eval:
        trainer.train()
        trainer.evaluate()
    else:
        trainer.load_model()
        trainer.evaluate()
