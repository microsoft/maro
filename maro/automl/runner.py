# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
import yaml
from redis import Redis

from maro.utils.logger import CliLogger
from .dispatcher import Dispatcher

'''
from nni.tuner import Tuner
from nni.algorithms.hpo.batch_tuner import BatchTuner
from nni.algorithms.hpo.evolution_tuner import EvolutionTuner
from nni.algorithms.hpo.gp_tuner import GPTuner
from nni.algorithms.hpo.gridsearch_tuner import GridSearchTuner
from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner
from nni.algorithms.hpo.metis_tuner import MetisTuner
from nni.algorithms.hpo.networkmorphism_tuner import NetworkMorphismTuner
from nni.algorithms.hpo.pbt_tuner import PBTTuner
from nni.algorithms.hpo.ppo_tuner import PPOTuner
from nni.algorithms.hpo.regularized_evolution_tuner import RegularizedEvolutionTuner
from nni.algorithms.hpo.smac_tuner import SMACTuner
'''
logger = CliLogger(name=__name__)

def choose_tuner(tuner_name: str, tuner_args: dict):
    if tuner_name == 'GridSearch':
        from nni.algorithms.hpo.gridsearch_tuner import GridSearchTuner
        return GridSearchTuner(**tuner_args)
    elif tuner_name == 'TPE':
        from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner
        return HyperoptTuner('tpe', **tuner_args)
    elif tuner_name == 'Random':
        from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner
        return HyperoptTuner('random_search', **tuner_args)
    elif tuner_name == 'Anneal':
        from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner
        return HyperoptTuner('anneal', **tuner_args)


if __name__ == "__main__":
    logger.info('starting runer...')
    with open('tuner_config.yml', 'r') as f:
        tuner_config = yaml.safe_load(f)

    tuner_name = tuner_config['nniTuner']['name']
    tuner_args = tuner_config['nniTuner']['classArgs']
    search_space_path = tuner_config['search_space']
    tuner_job_name = tuner_config['tuner_job_name']
    cluster_name = tuner_config['cluster_name']
    job_temp = tuner_config['job_temp']

    redis_host = tuner_config['redis_host']
    redis_port = tuner_config['redis_port']

    logger.info('starting tuner...')
    tuner = choose_tuner(tuner_name, tuner_args)

    with open(search_space_path, 'r') as fr:
        search_space = json.load(fr)

    tuner.update_search_space(search_space=search_space)

    dispatcher = Dispatcher(tuner, redis_host, redis_port, tuner_job_name, cluster_name, job_temp)
    logger.info('starting dispatcher...')
    dispatcher.run()
