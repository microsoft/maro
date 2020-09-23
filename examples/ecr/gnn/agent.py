from maro.rl import AbsAgent
from maro.utils import DummyLogger
import time
from collections import defaultdict
from examples.ecr.gnn.numpy_store import Shuffler
import numpy as np

class TrainableAgent(AbsAgent):
    def __init__(self, name, algorithm, experience_pool, logger=DummyLogger()):
        self._logger = logger
        super().__init__(name, algorithm, experience_pool)

    def train(self, training_config):
        loss_dict = defaultdict(list)
        loss_rt = defaultdict(float)
        # for code, exp_pool in experience_pool_dict.items():
        for j in range(training_config.shuffle_time):
            shuffler = Shuffler(self._experience_pool, batch_size=training_config.batch_size)
            while shuffler.has_next():
                batch = shuffler.next()
                actor_loss, critic_loss, entropy_loss, tot_loss = self._algorithm.train(batch, self._name[0], 
                        self._name[1])
                loss_dict['actor'].append(actor_loss)
                loss_dict['critic'].append(critic_loss)
                loss_dict['entropy'].append(entropy_loss)
                loss_dict['tot'].append(tot_loss)

        a_loss = np.mean(loss_dict['actor'])
        c_loss = np.mean(loss_dict['critic'])
        e_loss = np.mean(loss_dict['entropy'])
        tot_loss = np.mean(loss_dict['tot'])
        self._logger.debug('code: %s \t actor: %f \t critic: %f \t entropy: %f \t tot: %f'%(str(self._name), a_loss, 
                            c_loss, e_loss, tot_loss))
    
        self._experience_pool.clear()
        return loss_dict
    
    def choose_action(self, model_state):
        return self._algorithm.choose_action(model_state, self._name[0], self._name[1])
    