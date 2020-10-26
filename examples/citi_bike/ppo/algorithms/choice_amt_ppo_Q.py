from copy import deepcopy
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from examples.citi_bike.ppo.models.homo_gnn import STGNNBackend,TwoHeadLocalQ
from examples.citi_bike.ppo.models.separated_tpsc import AttTransPolicy
from examples.citi_bike.ppo.utils import from_numpy
from maro.utils import Logger, LogFormat

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
        self.temporal_gnn = STGNNBackend(node_dim + 6, out_dim=self.emb_dim, channel_cnt=self.channel_cnt)
        self.policy = AttTransPolicy(self.emb_dim, self.neighbor_cnt, graph_size)
        self.Q = TwoHeadLocalQ(self.emb_dim, graph_size, self.neighbor_cnt, activation_func=nn.ReLU)
        self._logger = Logger(tag="model", format_=LogFormat.simple,
                              dump_folder=log_pth, dump_mode="w", auto_timestamp=False)

        ts_path = kargs["ts_path"]
        self.writer = SummaryWriter(ts_path)

        self.old_policy = deepcopy(self.policy)
        self.old_temporal_gnn = deepcopy(self.temporal_gnn)
        self.old_Q = deepcopy(self.Q)

        self.policy = self.policy.to(device=self.device)
        self.old_policy = self.old_policy.to(device=self.device)
        self.old_policy.eval()
        self.temporal_gnn = self.temporal_gnn.to(device=self.device)
        self.old_temporal_gnn = self.old_temporal_gnn.to(device=self.device)
        self.old_temporal_gnn.eval()
        self.Q = self.Q.to(device=self.device)
        self.old_Q = self.old_Q.to(device=self.device)
        self.old_Q.eval()

        # optimizer
        self.temporal_gnn_opt = optim.Adam(self.temporal_gnn.parameters(), lr=3e-4)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.Q_opt = optim.Adam(self.Q.parameters(), lr=3e-4)

        # scheduler
        self.gnn_scheduler = StepLR(self.temporal_gnn_opt, step_size=100, gamma=0.5)
        self.policy_scheduler = StepLR(self.policy_opt, step_size=100, gamma=0.5)
        self.Q_scheduler = StepLR(self.Q_opt, step_size=100, gamma=0.5)

        # loss
        self.mse_loss = nn.MSELoss()
        self.K_epochs = 4
        self.eps_clip = 0.2
        self.amt_bucket = 30

    def obs_to_torch(self, obs):
        x = from_numpy(torch.FloatTensor, self.device, obs["x"][0])[0]
        time = from_numpy(torch.LongTensor, self.device, obs["x"][1])[0]
        edge_idx_list = from_numpy(torch.LongTensor, self.device, *obs["edge_idx_list"])
        action_edge_idx = from_numpy(torch.LongTensor, self.device, obs["action_edge_idx"])[0]
        acting_node = from_numpy(torch.LongTensor, self.device, obs["acting_node_idx"])[0]

        actual_amount = torch.FloatTensor(obs["actual_amount"]).to(device=self.device)
        return x, time, edge_idx_list, action_edge_idx, actual_amount, acting_node

    def batchize_obs(self, obs_list):
        batch_size = len(obs_list)
        idx_inc = np.arange(batch_size)*self.per_graph_size

        acting_node_idx = np.hstack([e["acting_node_idx"] for e in obs_list]) + idx_inc
        actual_amount = np.hstack([e["actual_amount"] for e in obs_list])
        action_edge_idx = np.hstack([obs_list[i]["action_edge_idx"]+idx_inc[i] for i in range(batch_size)])

        x = np.concatenate([e["x"][0] for e in obs_list], axis=1)
        time = np.concatenate([e["x"][1] for e in obs_list], axis=1)

        channel_cnt = len(obs_list[0]["edge_idx_list"])
        edge_idx_list = [np.hstack([obs_list[i]["edge_idx_list"][j] + idx_inc[i] for i in range(batch_size)])
                         for j in range(channel_cnt)]

        return {
            "acting_node_idx": acting_node_idx,
            "x": (x, time),
            "edge_idx_list": edge_idx_list,
            "action_edge_idx": action_edge_idx,
            "actual_amount": actual_amount,
        }

    def batchize_exp(self, exp_list):
        if not exp_list:
            return {}

        a = np.hstack([e["a"] for e in exp_list])

        # state
        s = self.batchize_obs([e["obs"] for e in exp_list])
        s_ = self.batchize_obs([e["obs_"] for e in exp_list])
        r = np.hstack([np.array(e["r"]) for e in exp_list])

        gamma = np.hstack([np.array(e["gamma"]) for e in exp_list])

        choice_prob = [e["supplement"]["choice_prob"] for e in exp_list]
        choice_prob = np.hstack(choice_prob)
        choice_idx = [e["supplement"]["choice_idx"] for e in exp_list]
        choice_idx = np.hstack(choice_idx)
        amt_prob = [e["supplement"]["amt_prob"] for e in exp_list]
        amt_prob = np.hstack(amt_prob)
        amt_idx = [e["supplement"]["amt_idx"] for e in exp_list]
        amt_idx = np.hstack(amt_idx)

        rlt = {
            "a": a,
            "s": s,
            "s_": s_,
            "r": r,
            "gamma": gamma,
            "choice_prob": choice_prob,
            "choice_idx": choice_idx,
            "amt_prob": amt_prob,
            "amt_idx": amt_idx
        }
        # supplement is handled by each algorithm (like GnnddPG), rather than outside.

        return rlt

    def act(self, obs):
        '''
        forward gnn and policy to get action
        '''
        with torch.no_grad():
            x, time_feature, edge_idx_list, action_edge_idx, actual_amount, acting_node = self.obs_to_torch(obs)
            x = torch.cat((x, time_feature.float()), -1)
            actual_amount = actual_amount.reshape(-1, self.neighbor_cnt)
            # emb = self.old_temporal_gnn(x, edge_idx_list[0], time_feature)
            edge = torch.cat((action_edge_idx[0, :-1].reshape(1, -1), action_edge_idx[1, :-1].reshape(1, -1)), 0)
            emb = self.old_temporal_gnn(x, edge)
            choice, att = self.old_policy.choose_destination(emb, action_edge_idx, actual_amount, acting_node)
            abs_choice = action_edge_idx[1, choice]
            amt_choice, amt_att = self.old_policy.determine_amount(emb, actual_amount, acting_node, abs_choice)
            batch_idx = torch.arange(choice.shape[0], dtype=torch.long).to(self.device)
            cnt = amt_choice * self.old_policy.amt_step * actual_amount[batch_idx, choice]
            return choice.cpu().numpy(), \
                cnt.cpu().numpy(), \
                {
                    "choice_idx": choice.cpu().numpy(),
                    "amt_idx": amt_choice.cpu().numpy(),
                    "choice_att": att,
                    "amt_att": amt_att,
                    "choice_prob": att[batch_idx, choice].cpu().numpy(),
                    "amt_prob": amt_att[batch_idx, amt_choice].cpu().numpy(),
                    'src': acting_node.cpu().numpy(),
                    'dest': abs_choice.cpu().numpy()}

    def grad(self, batch):
        global epoch_count
        global itr_count
        # Monte Carlo estimate of state rewards:
        batch_size = len(batch)
        batch = self.batchize_exp(batch)
        rewards = from_numpy(torch.FloatTensor, self.device, batch["r"])[0].reshape(-1, self.per_graph_size)
        # print("rewards shape", rewards.shape)
        rewards = rewards.float()

        self.writer.add_scalar("Reward\\", rewards.mean(), epoch_count)
        tot_gamma = from_numpy(torch.FloatTensor, self.device, batch["gamma"])[0]

        # normalize to a reasonable scope
        gamma = tot_gamma.reshape(-1, 1).repeat(1, self.per_graph_size)

        x, time, edge_idx_list, action_edge_idx, actual_amount, acting_node = self.obs_to_torch(batch["s"])
        x_, time_, edge_idx_list_, action_edge_idx_, actual_amount_, _ = self.obs_to_torch(batch["s_"])

        x = torch.cat((x, time.float()), -1)
        x_ = torch.cat((x_, time.float()), -1)

        # convert list to tensor
        # old_actions = from_numpy(torch.FloatTensor, self.device, batch["a"])[0]
        batch_arange = torch.arange(batch_size).to(device=self.device)
        old_choice, old_amt = from_numpy(torch.LongTensor, self.device, batch["choice_idx"], batch["amt_idx"])
        old_choice_prob, old_amt_prob = from_numpy(torch.FloatTensor, self.device, batch["choice_prob"],
                                                   batch["amt_prob"])
        old_action_prob = old_choice_prob * old_amt_prob
        loss_ret = []

        edge = torch.cat((action_edge_idx[0, :-1].reshape(1, -1), action_edge_idx[1, :-1].reshape(1, -1)), 0)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # def evaluate(self, obs, mask, actions):
            ts_emb = self.temporal_gnn(x, edge)
            # action_p is a tuple
            _, choice_att = self.policy.choose_destination(ts_emb, action_edge_idx, actual_amount, acting_node, sample=False)
            amt_choice, amt_att = self.policy.determine_amount(ts_emb, actual_amount, acting_node, old_choice, sample=True)

            choice_prob = choice_att[batch_arange, old_choice]
            amt_prob = amt_att[batch_arange, old_amt]
            action_prob = choice_prob * amt_prob

            choice_entropy = Categorical(probs=choice_att).entropy()
            amt_entropy = Categorical(probs=amt_att).entropy()

            real_amt = amt_choice.float().repeat(self.neighbor_cnt)
            action_p = torch.stack((choice_att.reshape(-1), real_amt.reshape(-1)), dim=-1).detach()
            _, state_values = self.Q(ts_emb, action_p, action_edge_idx)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = (action_prob + 0.00001) / (old_action_prob + 0.00001)
            ratios_choice = (choice_prob + 0.00001) / (old_choice_prob + 0.00001)
            ratios_amt = (amt_prob + 0.00001) / (old_amt_prob + 0.00001)
            # Finding Surrogate Loss:
            rewards = rewards.sum(-1)
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages

            surr2 = torch.clamp(ratios_choice, 1 - self.eps_clip, 1 + self.eps_clip) *\
                torch.clamp(ratios_amt, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            ploss = -torch.min(surr1, surr2)
            mloss = self.mse_loss(state_values, rewards)
            loss = ploss + 1000 * mloss - 0.1 * (amt_entropy+choice_entropy)

            print("mse loss", mloss.mean())
            self.writer.add_scalar("policy loss\\", ploss.mean(), itr_count)
            self.writer.add_scalar("mse loss\\", mloss.mean(), itr_count)
            self.writer.add_scalar("amt entropy\\", amt_entropy.mean(), itr_count)
            self.writer.add_scalar("choice entropy\\", choice_entropy.mean(), itr_count)
            self.writer.add_scalar("amt entropy std\\", amt_entropy.std(), itr_count)
            self.writer.add_scalar("choice entropy std\\", choice_entropy.std(), itr_count)

            # take gradient step
            self.temporal_gnn_opt.zero_grad()
            self.policy_opt.zero_grad()
            loss.mean().backward()
            # grad clip
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.temporal_gnn.parameters(), 0.5)

            self.temporal_gnn_opt.step()
            self.policy_opt.step()
            self.policy_scheduler.step()
            self.gnn_scheduler.step()

            loss_ret.append(loss.mean())

            itr_count += 1

        self.old_policy.load_state_dict(self.policy.state_dict())
        self.old_temporal_gnn.load_state_dict(self.temporal_gnn.state_dict())
        self.writer.add_scalar("Loss\\", sum(loss_ret)/len(loss_ret), epoch_count)
        epoch_count += 1

    def save(self, pth):
        torch.save([self.temporal_gnn.state_dict(), self.policy.state_dict()], pth)
