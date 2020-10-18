import os
import yaml

from torch.utils.data import RandomSampler, BatchSampler
from datetime import datetime
import pickle

# from examples.citi_bike.ppo.algorithms.choice_amt_ppo import AttGnnPPO
from examples.citi_bike.ppo.algorithms.fixamt_ppo_att import AttGnnPPO
from examples.citi_bike.ppo.actors.one_dest_actor import CitiBikeActor
# from examples.citi_bike.ppo.citibike_state_shaping import CitibikeStateShaping
from examples.citi_bike.ppo.fixamt_state_shaping import CitibikeStateShaping
# from examples.citi_bike.ppo.choice_amt_state_shaping import CitibikeStateShaping
from examples.citi_bike.ppo.post.truncated_reward import PostProcessor
# from examples.citi_bike.ppo.post.selfdef_reward import PostProcessor
from examples.citi_bike.ppo.utils import backup
from maro.simulator import Env
from maro.utils import Logger, LogFormat, convert_dottable


class CitiBikeLearner:
    def __init__(self):
        config_pth = os.environ.get('CONFIG') or os.path.join(os.getcwd(), 'examples/citi_bike/ppo/config/config.yml')

        with open(config_pth, 'r') as in_file:
            raw_config = yaml.safe_load(in_file)
            config = convert_dottable(raw_config)

        exp_name = config.experiment_name
        exp_name = '%s_%s' % (datetime.now().strftime('%H_%M_%S'), exp_name)
        exp_name_par = f"{datetime.now().strftime('%Y%m%d')}"
        log_folder = os.path.join(os.getcwd(), 'log', exp_name_par, exp_name)

        tensorboard_folder_train = os.path.join(os.getcwd(), 'log', 'train')
        if not os.path.exists(tensorboard_folder_train):
            os.makedirs(tensorboard_folder_train)
        tensorboard_folder_reward = os.path.join(os.getcwd(), 'log', 'reward')
        if not os.path.exists(tensorboard_folder_reward):
            os.makedirs(tensorboard_folder_reward)
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        with open(os.path.join(log_folder, 'config.yml'), 'w', encoding='utf8') as out_file:
            yaml.safe_dump(raw_config, out_file)
        # backup(os.path.join(os.getcwd(), 'examples/citi_bike/enc_gat'), os.path.join(log_folder, 'code/'))

        self.log_folder = log_folder
        print("log_folder is at ", self.log_folder)
        self._logger = Logger(tag='learner', format_=LogFormat.simple,
                                  dump_folder=self.log_folder, dump_mode='w', auto_timestamp=False)
        self.model_save_folder = self.log_folder + "/models"
        if not os.path.exists(self.model_save_folder):
            os.makedirs(self.model_save_folder)

        # self.rollouter = CitiBikeVecActor(CitibikeStateShaping, log_folder, batch_num=6)
        self.rollouter = CitiBikeActor(config.env, CitibikeStateShaping, PostProcessor, log_folder,
                                       ts_path=os.path.join(tensorboard_folder_reward,
                                                            "%s_%s" % (exp_name_par, exp_name)))
        # self.rollouter = ZeroActionActor(scenario=config.env.scenario, topology=config.env.topology,
        #                                  start_tick=config.env.start_tick, durations=config.env.durations,
        #                                  snapshot_resolution=config.env.snapshot_resolution)
        self.demo_env = Env(scenario=config.env.scenario, topology=config.env.topology,
                            start_tick=config.env.start_tick, durations=config.env.durations,
                            snapshot_resolution=config.env.snapshot_resolution)
        self.demo_state_shaping = CitibikeStateShaping(self.demo_env)

        station_cnt = len(self.demo_env.snapshot_list['stations'])
        channel_cnt = self.demo_state_shaping.channel_cnt
        reward, decision_evt, is_done = self.demo_env.step(None)
        neighbor_cnt = len(decision_evt.action_scope)-1
        # algorithm parameter
        algoirthm_config = {
            'emb_dim': config.model.emb_dim,
            'neighbor_cnt': neighbor_cnt,
            'gamma': config.train.gamma,
            'device': config.model.device,
            'ts_path': os.path.join(tensorboard_folder_train, "%s_%s" % (exp_name_par, exp_name)),
        }

        self.algorithm = AttGnnPPO(node_dim=self.demo_state_shaping.node_attr_len, channel_cnt=channel_cnt,
                                   graph_size=station_cnt, log_pth=log_folder, **algoirthm_config)
        self.config = config
        self.ts_path = os.path.join(tensorboard_folder_reward, "%s_%s" % (exp_name_par, exp_name))
        self.log_folder = log_folder

    def _save_code(self):
        src_pth = os.path.join(os.getcwd(), 'examples/citi_bike/ppo/')
        dest_pth = os.path.join(self.log_folder, 'code/')
        backup(src_pth, dest_pth)

    def train(self):
        rollout_cnt = self.config.train.rollout_cnt
        batch_size = self.config.train.batch_size
        flush_cnt = self.config.train.exp_replay.flush
        train_freq = self.config.train.freq
        stats_save_freq = 5

        exp_pool = []
        for i in range(rollout_cnt):
            # prediction
            if self.config.save_code_after == i+1:
                self._save_code()

            self._logger.debug(f'rollout cnt: {i}')
            is_save_log = i % stats_save_freq == stats_save_freq-1
            exp_list, stats = self.rollouter.sample(self.algorithm, save_log=is_save_log)

            if is_save_log:
                self.save_log(i, stats)

            for exp in exp_list:
                if exp is not None:
                    exp_pool.extend(exp)

            if i % train_freq == train_freq - 1:
                sampler = BatchSampler(RandomSampler(exp_pool), batch_size=batch_size, drop_last=False)
                for batch in sampler:
                    self.algorithm.grad([exp_pool[bid] for bid in batch])
            if i % flush_cnt == flush_cnt -1:
                exp_pool = []
            if i % 500 == 499:
                pth = os.path.join(self.log_folder, "nn_%d.pickle" % i)
                self.algorithm.save(pth)

    def save_log(self, itr, stats):
        with open(os.path.join(self.log_folder, 'stats_%d' % itr), 'wb') as fp:
            pickle.dump(stats, fp)


if __name__ == "__main__":
    learner = CitiBikeLearner()
    learner.train()
