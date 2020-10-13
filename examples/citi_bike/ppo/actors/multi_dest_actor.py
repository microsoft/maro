import numpy as np
import pandas as pd
import os
import yaml
import torch
from datetime import datetime
from maro.simulator.scenarios.citibike.common import Action, DecisionEvent, DecisionType
# from maro.simulator.scenarios.citibike.business_engine import BikeEventType
import maro.simulator.utils.random as sim_random
from maro.simulator import Env
from maro.utils import Logger, LogFormat, convert_dottable
from torch.utils.data import RandomSampler, BatchSampler
from maro.vector_env.vector_env import VectorEnv
from examples.citi_bike.enc_gat.post.overall_return import PostProcessor
from examples.citi_bike.enc_gat.utils import batchize
import pickle


class CitiBikeActor:
    def __init__(self, state_shaping_cls, log_folder):
        self._logger = Logger(tag='actor', format_=LogFormat.simple,
                                  dump_folder=log_folder, dump_mode='w', auto_timestamp=False)
        
        #simulator_logger = Logger(tag='simulator', format_=LogFormat.simple,
        #                            dump_folder=log_folder, dump_mode='w', auto_timestamp=False)

        self.log_folder = log_folder
        self.env = Env(scenario='citibike', topology='train', start_tick=0, durations=1440, snapshot_resolution=60)
        self.state_shaping = state_shaping_cls(self.env)
        self.post_processing = PostProcessor(self.env)

    def sample(self, policy):
        reward, decision_evt, is_done = self.env.step(None)
        stats = []
        acts = []
        while not is_done:
            obs = self.state_shaping.get_states(reward, decision_evt)
            self.post_processing.record(decision_event=decision_evt, obs=obs)
            # batch_obs = batchize(states)
            
            stats.append([(obs['tick'], obs['shortage'], obs['fulfillment']),])
            nums, neighbor_idxes = policy.act(obs)
            acts.append(nums)
            
            actions = []
            station_idx = decision_evt.station_idx
            nums /= self.state_shaping.action_scaler
            action_cnt = nums.shape[-1]
            for i in range(action_cnt):
                act = nums[0, i]
                if act > 0:
                    actions.append(Action(station_idx, neighbor_idxes[1, 0, i], int(abs(act))))
                else:
                    actions.append(Action(neighbor_idxes[1, 0, i], station_idx, int(abs(act))))
                    
            self.post_processing.record(action=actions)
            reward, decision_evt, is_done = self.env.step(actions)
        trajectories = self.post_processing()
        self.env.reset()
        self.post_processing.reset()
        self._logger.debug('Average shortage & fulfillment: %f, %f'% self.compute_final_shortage(stats))

        return [trajectories,], stats 

    def compute_final_shortage(self, stats):
        sh = [np.sum(s[1]) for s in stats[-1]]
        fl = [np.sum(s[2]) for s in stats[-1]]
        return np.mean(sh), np.mean(fl)


class CitiBikeVecActor:
    def __init__(self, state_shaping_cls, log_folder, batch_num=2):
        self._logger = Logger(tag='actor', format_=LogFormat.simple,
                                  dump_folder=log_folder, dump_mode='w', auto_timestamp=False)
        
        #simulator_logger = Logger(tag='simulator', format_=LogFormat.simple,
        #                            dump_folder=log_folder, dump_mode='w', auto_timestamp=False)

        self.log_folder = log_folder
        self.batch_num = batch_num
        self.env = VectorEnv(batch_num=batch_num, states_extractor_cls=state_shaping_cls,
                    post_processor_cls=PostProcessor,
                   scenario='citibike', topology='train', start_tick=0, durations=1000, snapshot_resolution=60)
        # self.env = Env(scenario='citibike', topology='train', start_tick=0, durations=1440, snapshot_resolution=60)

    def sample(self, policy):
        reward, decision_evt, is_done = self.env.step(None)
        stats = []
        while not is_done:
            states = self.env.states
            batch_obs = batchize(states)
            
            stats.append([(s['tick'], s['shortage'], s['fulfillment']) for s in states])
            nums, neighbor_idxes = policy.act(batch_obs)
            
            actions = []
            batch_size = nums.shape[0]
            for i in range(batch_size):
                action = []
                station_idx = decision_evt[i]['station_idx']
                moves, nidxes = nums[i], neighbor_idxes[1, i]
                
                for m, n in zip(moves, nidxes):
                    act_m = int(abs(m*10))
                    '''
                    if act_m == 0:
                        continue
                    '''
                    if m > 0:
                        action.append(Action(station_idx, n, act_m))
                    else:
                        action.append(Action(n, station_idx, act_m))
                actions.append(action)
                    
            reward, decision_evt, is_done = self.env.step(actions)
        self.env.reset()
        self._logger.debug('Average shortage & fulfillment: %f, %f'% self.compute_final_shortage(stats))

        return self.env.trajectories, stats 

    def compute_final_shortage(self, stats):
        sh = [np.sum(s[1]) for s in stats[-1]]
        fl = [np.sum(s[2]) for s in stats[-1]]
        return np.mean(sh), np.mean(fl)


if __name__ == "__main__":
    log_folder = r'/home/wenshi/maro_internal/examples/citibike/log_tmp'
    cba = CitiBikeActor(log_folder)
    cba.sample(None)