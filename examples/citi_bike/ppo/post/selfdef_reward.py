import numpy as np

class PostProcessor:
    def __init__(
        self,
        env,
        transfer_cost=0.02,
        gamma=0.9,
        delay_length=5,
        max_delay_length=100,
        fixed_window_size=False,
        state_shaping=None,
        foresee=3
    ):
        self.env = env
        self.station_cnt = len(self.env.snapshot_list['stations'])
        self.trajectory = []

        self.hist_obs = {}
        self.latest_obs_frame_index = -1
        self.cur_frame_index = 0
        self.cur_obs, self.cur_action = None, None

        self.transfer_cost = transfer_cost
        self.delay_length = delay_length
        self.foresee = foresee
        self.max_delay_length = max_delay_length
        self.decay_factors = np.logspace(0, max_delay_length + 1, max_delay_length + 1, base=gamma)

        self.fixed_window_size = fixed_window_size
        self.state_shaping = state_shaping
        if self.fixed_window_size and self.state_shaping == None:
            print("State-Shaping instance is required in fixed window size mode!")
            exit(0)
        self.norm_ratio = 0.2
        reward_name = "Fixed Truncated Reward" if self.fixed_window_size else "Truncated Reward"
        print(f"*********** {reward_name} ***********")

    def __call__(self):
        order_data = self.env.snapshot_list['stations'][
                : :['fulfillment', 'shortage','min_bikes']
            ].reshape(-1, self.station_cnt, 3)
        # reward_per_frame = order_data[:, :, 0] - order_data[:, :, 1]
        reward_per_frame = - order_data[:, :, 1]
        inventary_per_frame = order_data[:,:,2]
        shortage_per_frame = order_data[:,:,1]

        rm_list = []
        for i, sars in enumerate(self.trajectory):
            if sars['frame_index'] >= self.latest_obs_frame_index:
                self.trajectory = self.trajectory[:i]
                break

            end_index = sars['frame_index'] + self.delay_length
            end_foresee = sars['frame_index'] + self.foresee
            # print("frame_idx",sars['frame_index'])
            if self.fixed_window_size:
                if end_index > self.env.frame_index:
                    self.trajectory = self.trajectory[:i]
                    break
                if end_index not in self.hist_obs.keys():
                    self.hist_obs[end_index] = self.state_shaping.get_states(frame_index=end_index)
            else:
                max_index = min(self.env.frame_index, sars['frame_index'] + self.max_delay_length)
                while end_index not in self.hist_obs.keys() and end_index <= max_index:
                    end_index += 1
                if end_index > max_index:
                    if self.latest_obs_frame_index < max_index:
                        end_index = self.latest_obs_frame_index
                    else:
                        rm_list.append(i)
                        continue

            sars['obs_'] = self.hist_obs[end_index]
            sars['gamma'] = self.decay_factors[end_index - sars['frame_index']]

            amount = np.abs(sars['a'][1])
            amount_cost = amount*self.transfer_cost
            sars['r'] = np.dot(
                self.decay_factors[:end_index - sars['frame_index']],
                reward_per_frame[sars['frame_index'] + 1:end_index + 1]
            )
            acting_node = sars['obs']['acting_node_idx']
            sars['r'][acting_node] -= amount_cost
            # # use shortage for src and dest
            # neighbor_idx,amt = sars['a']
            # neighbor_idx,amt = int(neighbor_idx[0]),int(amt[0]/0.1)
            # act_idx = sars['obs']['acting_node_idx']
            # _, col = sars['obs']['action_edge_idx']
            # target_idx = col[neighbor_idx]
            # sars['self_r'] = sars['r'][col].sum()

            #  self define the reward with trace info
            sars['self_r'] = 0
            neighbor_idx,amt = sars['a']
            neighbor_idx,amt = int(neighbor_idx[0]),int(amt[0]/0.1)
            act_idx = sars['obs']['acting_node_idx']
            _, col = sars['obs']['action_edge_idx']
            target_idx = col[neighbor_idx]
            if(amt>0 and act_idx != target_idx):
                # supply: act_idx -> target_idx
                max_shortage = np.amax(shortage_per_frame[sars['frame_index']+1:end_foresee+1,act_idx])
                min_invt = np.amin(inventary_per_frame[sars['frame_index'] + 1:end_foresee + 1,target_idx])
                shortage_created = min(amt,max_shortage)
                shortage_reduced = max(amt-min_invt,0)
                # # refine on target
                # if(max_shortage > 0):
                #     sars['self_r'] = -1
                # elif(amt > min_invt):
                #     sars['self_r'] = 1
                sars['self_r'] = self.norm_ratio*(shortage_reduced-shortage_created)
            elif(amt<0 and act_idx != target_idx):
                # demand: act_idx <- target_idx
                max_shortage = np.amax(shortage_per_frame[sars['frame_index'] + 1:end_foresee + 1,target_idx])
                min_invt = np.amin(inventary_per_frame[sars['frame_index'] + 1:end_foresee + 1,act_idx])
                shortage_created = min(amt,max_shortage)
                shortage_reduced = max(amt-min_invt,0)
                # # refine on target
                # if(max_shortage > 0):
                #     sars['self_r'] = -1
                # elif(abs(amt) > min_invt):
                #     sars['self_r'] = 1
                sars['self_r'] = self.norm_ratio*(shortage_reduced-shortage_created)

        for i in rm_list[::-1]:
            self.trajectory.pop(i)

        return self.trajectory

    def record(self, decision_event=None, obs=None, action=None):
        if decision_event is not None and obs is not None:
            self.cur_obs = obs
            frame_index = decision_event.frame_index
            self.cur_frame_index = frame_index

            self.hist_obs[frame_index] = obs
            self.latest_obs_frame_index = frame_index

        if action is not None:
            self.cur_action = action

        if self.cur_obs is None or self.cur_action is None:
            return

        choice, amount, record_data = self.cur_action

        self.trajectory.append({
            'obs': self.cur_obs,
            'a': np.array([choice, amount]),
            'obs_': None,
            'r': None,
            'self_r': None,
            'gamma': np.ones(self.station_cnt),
            'frame_index': self.cur_frame_index,
            'supplement': record_data,
        })
        self.cur_obs, self.cur_action = None, None


    def reset(self):
        self.trajectory = []
        self.hist_obs = {}
        self.latest_obs_frame_index = -1
        self.cur_obs, self.cur_action = None, None

