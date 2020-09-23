from maro.rl import AbsLearner
from maro.rl import AbsActor
from maro.rl import AbsAgentManager
from maro.utils import DummyLogger
import time
import os

class GNNLearner(AbsLearner):
    def __init__(self, actor: AbsActor, trainable_agents:AbsAgentManager, logger=DummyLogger()):
        """Learner class for the training pipeline and the specialized logging in GNN solution for CIM problem.
        
        Args:
            actor (AbsActor): The actor instance to collect experience.
            trainable_agents (AbsAgentManager): The agent manager for training RL models.
            logger (Logger)
        """
        self._actor = actor
        self._trainable_agents = trainable_agents
        self._logger = logger

    def train(self, training_config, log_pth=None):
        rollout_time = 0
        training_time = 0
        for i in range(training_config.rollout_cnt):
            self._logger.info('rollout %d'%(i+1))
            tick = time.time()
            exp_dict = self._actor.roll_out()

            rollout_time += time.time() - tick

            self._logger.info('start putting exps')
            self._trainable_agents.store_experiences(exp_dict)

            if training_config.enable and i % training_config.train_freq == training_config.train_freq-1:
                tick = time.time()

                # for code, exp_pool in experience_pool_dict.items():
                loss_dict = self._trainable_agents.train(training_config)
                    
                training_time += time.time() - tick

            if log_pth is not None and (i+1) % training_config.model_save_freq == 0:
                self._actor.save_model(os.path.join(log_pth, 'models'), i+1)

            self._logger.debug('total rollout_time: %d' % int(rollout_time))
            self._logger.debug('train_time: %d' % int(training_time))


    def test(self):
        pass