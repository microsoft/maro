from collections import defaultdict

import numpy as np

from maro.rl import AbsAgent
from maro.utils import DummyLogger

from .numpy_store import Shuffler
from .utils import gnn_union


class GNNAgent(AbsAgent):
    def __init__(
        self, 
        name, 
        algorithm, 
        experience_pool, 
        num_batches, 
        batch_size,
        logger=DummyLogger()
    ):
        super().__init__(name, algorithm, experience_pool=experience_pool)
        self._num_batches = num_batches
        self._batch_size = batch_size
        self._logger = logger
    
    def train(self):
        loss_dict = defaultdict(list)
        for j in range(self._num_batches):
            shuffler = Shuffler(self._experience_pool, batch_size=self._batch_size)
            while shuffler.has_next():
                batch = shuffler.next()
                actor_loss, critic_loss, entropy_loss, tot_loss = self._algorithm.train(
                    batch["s"], batch["a"], batch["R"], batch["s_"], self._name[0], self._name[1]
                )
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
