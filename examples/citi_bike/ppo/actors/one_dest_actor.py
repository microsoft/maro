from torch.utils.tensorboard import SummaryWriter

from examples.citi_bike.ppo.utils import batch_split
# from examples.citi_bike.enc_gat.rule_action_shaping import ActionShaping
from examples.citi_bike.ppo.action_shaping import ActionShaping
# from maro.simulator.scenarios.citibike.business_engine import BikeEventType
from maro.simulator import Env
from maro.utils import Logger, LogFormat

epoch_count = 0


def evaluate(env):
    station_ss = env.snapshot_list['stations']
    shortage_states = station_ss[::'shortage']
    fulfillment_states = station_ss[::'fulfillment']
    shortage_by_station = shortage_states.reshape(-1, len(station_ss))
    fulfillment_by_station = fulfillment_states.reshape(-1, len(station_ss))
    return shortage_states.sum(), fulfillment_states.sum(), shortage_by_station, fulfillment_by_station


def listize_record(data_dict):
    tmp = {k: batch_split(v) for k, v in data_dict.items()}
    keys = list(tmp.keys())
    batch_size = len(tmp[keys[0]])
    rlt = []
    for i in range(batch_size):
        rlt.append({k: tmp[k][i] for k in keys})
    return rlt


class CitiBikeActor:
    def __init__(self, env_config, state_shaping_cls, post_processing_cls, log_folder, ts_path=None):
        self._logger = Logger(tag='actor', format_=LogFormat.simple,
                                  dump_folder=log_folder, dump_mode='w', auto_timestamp=False)
        # simulator_logger = Logger(tag='simulator', format_=LogFormat.simple,
        #                            dump_folder=log_folder, dump_mode='w', auto_timestamp=False)

        print("*********** CitiBikeActor ***********")

        self.log_folder = log_folder
        '''
        if ts_path is None:
            tensorboard_folder = self.log_folder + "/tensorboard"
            self.writer = SummaryWriter(tensorboard_folder+ "/citibike_reward_%s"%env_config.topology)
            if not os.path.exists(tensorboard_folder):
                os.makedirs(tensorboard_folder)
        else:
        '''
        self.writer = SummaryWriter(ts_path)
        self.env = Env(scenario=env_config.scenario, topology=env_config.topology, start_tick=env_config.start_tick, durations=env_config.durations, snapshot_resolution=env_config.snapshot_resolution)

        self.state_shaping = state_shaping_cls(self.env)
        self.post_processing = post_processing_cls(self.env, transfer_cost=env_config.transfer_cost)
        # self.post_processing = post_processing_cls(self.env, fixed_window_size=True, state_shaping=self.state_shaping)
        self.action_shaping = ActionShaping()

    def sample(self, policy, save_log=False):
        global epoch_count
        reward, decision_evt, is_done = self.env.step(None)
        action_history = []
        total_amt = 0
        total_trip = 0
        while not is_done:
            obs = self.state_shaping.get_states(reward, decision_evt)
            self.post_processing.record(decision_event=decision_evt, obs=obs)
            # batch_obs = batchize(states)

            choice, num, record_data = policy.act(obs)
            action = self.action_shaping(decision_evt, choice, num)
            # print(action.number, list(decision_evt.action_scope.values())[choice[0]])
            neighbors = list(decision_evt.action_scope.keys())[:-1]
            total_amt += action.number
            if(action.number > 0):
                total_trip += 1
            if save_log:
                action_history.append({
                    'tick': self.env.tick,
                    'choice_att': {n: float(record_data['choice_att'][0, i]) for i, n in enumerate(neighbors)},
                    'from': action.from_station_idx,
                    'to': action.to_station_idx,
                    'amt': action.number,
                    # 'amt_att': {i*0.1: float(record_data['amt_att'][0,i]) for i in range(11)}
                })

            reward, decision_evt, is_done = self.env.step(action)
            # reward, decision_evt, is_done = self.env.step([])

            self.post_processing.record(action=(choice, num, record_data))

        trajectories = self.post_processing()
        if save_log:
            reward_history = []
            for traj in trajectories:
                reward_history.append({
                    'fid': traj['frame_index'],
                    'r': traj['r']
                })
        shortage, fulfillment, shortage_by_stations, fulfillment_by_stations = evaluate(self.env)
        self._logger.debug(f'Average shortage & fulfillment: {int(shortage)}, {int(fulfillment)}')
        self.writer.add_scalar('Shortage\\', shortage, epoch_count)
        self.writer.add_scalar('Fulfillment\\', fulfillment, epoch_count)
        self.writer.add_scalar('Total amount\\', total_amt, epoch_count)
        self.writer.add_scalar('Total trip\\', total_trip, epoch_count)
        epoch_count += 1
        self.env.reset()
        self.post_processing.reset()

        if save_log:
            stats = {
                'fulfillment': fulfillment_by_stations,
                'shortage': shortage_by_stations,
                'actions': action_history,
                'reward': reward_history,
            }
            return [trajectories, ], stats
        else:
            return [trajectories, ], None


class ZeroActionActor:
    def __init__(self, scenario, topology, start_tick, durations, snapshot_resolution):
        self.env = Env(scenario=scenario, topology=topology, start_tick=start_tick, durations=durations, snapshot_resolution=snapshot_resolution)
        pass

    def sample(self):
        reward, decision_evt, is_done = self.env.step(None)
        while not is_done:
            reward, decision_evt, is_done = self.env.step([])
            # print("decision evt",decision_evt)
            # reward, decision_evt, is_done = self.env.step([])
        shortage, fulfillment, tot_shortage, tot_fulfillment = evaluate(self.env)
        # print(1.0*shortage/(shortage+fulfillment), shortage, fulfillment)
        self.env.reset()
        return shortage, fulfillment


if __name__ == "__main__":
    # evaluating_topo = ['c1', 'c2', 'c3', 'c4', 'region']
    evaluating_topo = ['c1']
    sample_cnt = 1
    durations = 28800

    for topo in evaluating_topo:

        print("Simulating: ", topo)
        tot_shortage, tot_fulfillment = 0, 0
        for _ in range(sample_cnt):
            actor = ZeroActionActor("citibike", topo, 1440, durations, 80)
            s, f = actor.sample()
            tot_shortage += s
            tot_fulfillment += f

        print(1.0*tot_shortage/(tot_shortage+tot_fulfillment), tot_shortage, tot_fulfillment)
