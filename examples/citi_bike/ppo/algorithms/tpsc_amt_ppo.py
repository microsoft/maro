import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
import numpy as np
import os
from examples.citi_bike.ppo.models.homo_gnn import GNNBackend
from examples.citi_bike.ppo.models.tpsc_amt import AttTransPolicy
from copy import deepcopy
from examples.citi_bike.ppo.utils import from_numpy, time_obs_to_torch
from maro.utils import Logger, LogFormat
from torch.utils.tensorboard import SummaryWriter

epoch_count = 0
itr_count = 0

class AttGnnPPO:
    def __init__(self, node_dim, channel_cnt, graph_size, log_pth, device="cuda:0", **kargs):
        self.device = torch.device(device)
        self.emb_dim = kargs["emb_dim"]
        self.channel_cnt = channel_cnt
        self.neighbor_cnt = kargs["neighbor_cnt"]
        self.per_graph_size = graph_size
        self.gamma = kargs["gamma"]
        # self.temporal_gnn = STGNNBackend(node_dim,out_dim=self.emb_dim,channel_cnt=self.channel_cnt)
        self.temporal_gnn = GNNBackend(time_window=20, node_dim=node_dim, output_dim=self.emb_dim,
                                       output_channels=self.channel_cnt)
        self.policy = AttTransPolicy(self.emb_dim, self.neighbor_cnt, graph_size)
        # self.gnn.load_state_dict(torch.load("/home/xiaoyuan/maro_internal/log/2020082111/0119_test/toy_gnn.pickle"))
        # self.policy = AttPolicy(self.emb_dim, self.neighbor_cnt, graph_size)
        self._logger = Logger(tag="model", format_=LogFormat.simple,
                              dump_folder=log_pth, dump_mode="w", auto_timestamp=False)
        tensorboard_pth = log_pth + "/tensorboard"
        if not os.path.exists(tensorboard_pth):
            os.makedirs(tensorboard_pth)
        self.writer = SummaryWriter(tensorboard_pth+ "/citibike_trans")

        self.old_policy = deepcopy(self.policy)
        self.old_temporal_gnn = deepcopy(self.temporal_gnn)

        self.policy = self.policy.to(device=self.device)
        self.old_policy = self.old_policy.to(device=self.device)
        self.old_policy.eval()
        self.temporal_gnn = self.temporal_gnn.to(device=self.device)
        self.old_temporal_gnn = self.old_temporal_gnn.to(device=self.device)
        self.old_temporal_gnn.eval()

        # optimizer
        self.temporal_gnn_opt = optim.Adam(self.temporal_gnn.parameters(),lr=3e-4)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=3e-4)

        # loss
        self.mse_loss = nn.MSELoss()
        self.K_epochs = 4
        self.eps_clip = 0.2
        self.amt_bucket = 30

    def act(self, obs):
        '''
        forward gnn and policy to get action
        '''
        with torch.no_grad():
            x, time_feature, edge_idx_list, action_edge_idx, actual_amount, per_graph_size = \
                time_obs_to_torch(obs, self.device)
            x = x.transpose(0, 1)
            time = [time_feature[:, :, i] for i in range(6)]
            edge = torch.cat((action_edge_idx[0, :-1].reshape(1, -1), action_edge_idx[1, :-1].reshape(1, -1)), 0)
            emb = self.old_temporal_gnn(x, edge, time)
            choice, att, amt_choice, amt_att = self.old_policy(emb, action_edge_idx, actual_amount)
            # print("att_dist",att)
            cnt = amt_choice * self.old_policy.amt_step
            batch_idx = torch.arange(choice.shape[0], dtype=torch.long).to(self.device)
            return choice.cpu().numpy(), \
                cnt.cpu().numpy(), \
                {"choice_att": att,
                 "amt_att": amt_att,
                 "choice_prob": att[batch_idx, choice].cpu().numpy(),
                 "amt_prob": amt_att[batch_idx, amt_choice].cpu().numpy()}

    def grad(self, batch):
        global epoch_count
        global itr_count
        # Monte Carlo estimate of state rewards:
        rewards = from_numpy(torch.FloatTensor, self.device, batch["r"])[0].reshape(-1, self.per_graph_size)
        rewards = rewards.float()
        # print("reward mean",rewards.mean())
        self.writer.add_scalar("Reward\\", rewards.mean(), epoch_count)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        tot_gamma = from_numpy(torch.FloatTensor, self.device, batch["gamma"])[0]

        # normalize to a reasonable scope
        gamma = tot_gamma.reshape(-1, 1).repeat(1, self.per_graph_size)
        # print("gamma",gamma)
        print(batch["s"])
        x, time_feature, edge_idx_list, action_edge_idx, actual_amount, per_graph_size =\
            time_obs_to_torch(batch["s"], self.device)
        x_, time_feature_, edge_idx_list_, action_edge_idx_, actual_amount_, per_graph_size_ =\
            time_obs_to_torch(batch["s_"], self.device)

        # convert list to tensor
        old_actions = from_numpy(torch.FloatTensor, self.device, batch["a"])[0]
        batch_arange = torch.arange(old_actions.shape[1]).to(device=self.device)
        old_choice, old_amt = old_actions[0].long(), (old_actions[1]/self.old_policy.amt_step).long()
        old_choice_prob, old_amt_prob = self.supplement2torch(batch["supplement"])
        old_action_prob = old_choice_prob.float() * old_amt_prob.float()
        loss_ret = []

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # def evaluate(self, obs, mask, actions):
            print("action edge idx shape", action_edge_idx.shape)
            # edge = torch.cat((action_edge_idx[0, :-1].reshape(1, -1), action_edge_idx[1, :-1].reshape(1, -1)), 0)
            ts_emb = self.temporal_gnn(x, edge_idx_list)
            ts_emb_ = self.old_temporal_gnn(x_, edge_idx_list_)
            # action_p is a tuple
            choice, choice_att, amt, amt_att = self.policy(ts_emb, action_edge_idx, actual_amount, old_choice)
            choice_prob = choice_att[batch_arange, old_choice]
            amt_prob = amt_att[batch_arange, old_amt]
            action_prob = choice_prob * amt_prob

            att_entropy = Categorical(probs=choice_att).entropy() + Categorical(probs=amt_att).entropy()

            state_values = self.policy.value(ts_emb)
            state_values_ = self.old_policy.value(ts_emb_).detach()

            # Finding the ratio (pi_theta / pi_theta__old):
            # ratios = torch.exp(action_logprobs - old_logprobs.detach())
            ratios = (action_prob+0.00001)/(old_action_prob+0.00001)
            # Finding Surrogate Loss:
            advantages = rewards + gamma*state_values_ - state_values.detach()
            advantages = advantages.sum(-1)
            surr1 = ratios * advantages.mean()
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages.mean()

            ploss = -torch.min(surr1, surr2)
            mloss = self.mse_loss(state_values, rewards+gamma*state_values_)
            loss = ploss + 100*mloss - 1.0*att_entropy
            # print("rewards",rewards)
            # print("state_values",state_values)
            # print("state_values_",state_values_)
            # print("minus",state_values-state_values_)
            # loss = mloss
            print("ratios", ratios.mean())
            print("advantage", advantages.mean())
            print("mse loss", mloss.mean())
            self.writer.add_scalar("policy loss\\", ploss.mean(), itr_count)
            self.writer.add_scalar("mse loss\\", mloss.mean(), itr_count)
            self.writer.add_scalar("entropy\\", att_entropy.mean(), itr_count)

            # take gradient step
            self.temporal_gnn_opt.zero_grad()
            self.policy_opt.zero_grad()
            loss.mean().backward()
            self.temporal_gnn_opt.step()
            self.policy_opt.step()

            loss_ret.append(loss.mean())

            itr_count += 1

        self.old_policy.load_state_dict(self.policy.state_dict())
        self.old_temporal_gnn.load_state_dict(self.temporal_gnn.state_dict())
        self.writer.add_scalar("Loss\\", sum(loss_ret)/len(loss_ret), epoch_count)
        epoch_count += 1

    def supplement2torch(self, sup):
        choice_prob = [sup_i["choice_prob"] for sup_i in sup]
        choice_prob = torch.from_numpy(np.vstack(choice_prob)).to(self.device)
        amt_prob = [sup_i["amt_prob"] for sup_i in sup]
        amt_prob = torch.from_numpy(np.vstack(amt_prob)).to(self.device)

        return choice_prob.reshape(-1), amt_prob.reshape(-1)

    def save(self, pth):
        torch.save([self.temporal_gnn, self.policy], pth)
