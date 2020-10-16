from collections import defaultdict

import numpy as np

from examples.cim.gnn.numpy_store import Shuffler
from maro.rl import AbsAgent
from maro.utils import DummyLogger


class TrainableAgent(AbsAgent):
    def __init__(self, name, algorithm, experience_pool, logger=DummyLogger()):
        self._logger = logger
        super().__init__(name, algorithm, experience_pool)

    def train(self, training_config):
        loss_dict = defaultdict(list)
        for j in range(training_config.shuffle_time):
            shuffler = Shuffler(self._experience_pool, batch_size=training_config.batch_size)
            while shuffler.has_next():
                batch = shuffler.next()
                actor_loss, critic_loss, entropy_loss, tot_loss = self._algorithm.train(
                    batch, self._name[0], self._name[1])
                loss_dict["actor"].append(actor_loss)
                loss_dict["critic"].append(critic_loss)
                loss_dict["entropy"].append(entropy_loss)
                loss_dict["tot"].append(tot_loss)

        a_loss = np.mean(loss_dict["actor"])
        c_loss = np.mean(loss_dict["critic"])
        e_loss = np.mean(loss_dict["entropy"])
        tot_loss = np.mean(loss_dict["tot"])
        self._logger.debug(
            f"code: {str(self._name)} \t actor: {float(a_loss)} \t critic: {float(c_loss)} \t entropy: {float(e_loss)} \
            \t tot: {float(tot_loss)}")

        self._experience_pool.clear()
        return loss_dict

    def choose_action(self, model_state):
        return self._algorithm.choose_action(model_state, self._name[0], self._name[1])
