import torch
from torch import nn
from torch import optim
import numpy as np
from examples.citi_bike.enc_gat.models.homo_gnn import GNNBackend, Policy, SparsePolicy, NaivePolicy, Q, TwoStepPolicy
from copy import deepcopy
from examples.citi_bike.enc_gat.utils import batchize, from_numpy, from_list, obs_to_torch, polyak_update, de_batchize
from itertools import chain
from maro.utils import Logger, LogFormat, convert_dottable

class GnnddPG:
    def __init__(self, node_dim, channel_cnt, graph_size, log_pth, device='cuda:1', **kargs):
        self.device = torch.device(device)
        self.emb_dim = kargs['emb_dim']
        self.channel_cnt = channel_cnt
        self.neighbor_cnt = kargs['neighbor_cnt']
        self.per_graph_size = graph_size
        self.gamma = kargs['gamma']
        self.gnn = GNNBackend(node_dim, out_dim=self.emb_dim, channel_cnt=self.channel_cnt)
        self.Q = Q(self.emb_dim, graph_size, activation_func=nn.ReLU)
        self.policy = Policy(self.emb_dim, self.neighbor_cnt, graph_size)

        self._logger = Logger(tag='model', format_=LogFormat.simple,
                                dump_folder=log_pth, dump_mode='w', auto_timestamp=False)

        self.target_gnn = deepcopy(self.gnn)
        self.target_policy = deepcopy(self.policy)
        self.target_Q = deepcopy(self.Q)

        self.gnn = self.gnn.to(device=self.device)
        self.target_gnn = self.target_gnn.to(device=self.device)
        self.Q = self.Q.to(device=self.device)
        self.target_Q = self.target_Q.to(device=self.device)
        self.policy = self.policy.to(device=self.device)
        self.target_policy = self.target_policy.to(device=self.device)

        # optimizer
        self.qs_opt = optim.Adam(chain(self.gnn.parameters(), self.Q.parameters()), lr=0.00005)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=0.00005)

        # loss
        self.qs_loss = nn.MSELoss()
        self.polyak = kargs['polyak']

    def act(self, obs):
        '''
        forward gnn and policy to get action
        '''
        with torch.no_grad():
            x, edge_idx_list, action_edge_idx, actual_amount, per_graph_size = obs_to_torch(obs, self.device)
            emb = self.gnn(x, edge_idx_list)
            action = self.policy(emb, action_edge_idx, actual_amount)
            atmp = action.cpu().numpy().reshape(-1, 21)
            aedge = de_batchize(action_edge_idx, self.per_graph_size).cpu().numpy().reshape(2, -1, 21)
            return atmp, aedge

    def grad(self, batch):
        '''
        forward gnn, policy and Q to get gradient
        '''
        # clean up grad
        self.gnn.zero_grad()
        self.Q.zero_grad()
        self.policy.zero_grad()

        x, edge_idx_list, action_edge_idx, actual_amount, per_graph_size = obs_to_torch(batch['s'], self.device)
        x_, edge_idx_list_, action_edge_idx_, actual_amount_, _ =  obs_to_torch(batch['s_'], self.device)
        actions, r, gamma = from_numpy(torch.FloatTensor, self.device, batch['a'], batch['r'], batch['gamma'])

        emb = self.gnn(x, edge_idx_list)
        # actions = self.target_policy(emb, action_edge_x, action_edge_index, actual_amount, per_graph_size)
        Qs = self.Q(emb, actions, action_edge_idx).reshape(-1)

        # state_
        '''
        emb_ = self.target_gnn(x_, edge_idx_list_)
        actions_ = self.target_policy(emb_, action_edge_idx_, actual_amount_)
        Qs_ = self.target_Q(emb_, actions_, action_edge_idx_).reshape(-1)
        '''

        # reward & gamma
        # Q loss
        # target_Qs = r + gamma*Qs_
        Qs_loss = self.qs_loss(r, Qs)
        qloss = Qs_loss.cpu().item()
        Qs_loss.backward(retain_graph=True)
        self.qs_opt.step()

        self.gnn.zero_grad()
        self.Q.zero_grad()
        # compute action from current policy
        emb = self.gnn(x, edge_idx_list)
        action_p = self.policy(emb, action_edge_idx, actual_amount)
        Qs_p = self.Q(emb, action_p, action_edge_idx)
        # policy loss
        policy_loss = -torch.mean(torch.sum(Qs_p, dim=1))
        ploss = policy_loss.cpu().item()
        policy_loss.backward()
        self.policy_opt.step()

        self._logger.debug('Q_loss: %f, p_loss: %f'%(qloss, ploss))
        # self._update_target()
        return qloss, ploss

    def _update_target(self):
        polyak_update(self.polyak, self.target_gnn, self.gnn)
        polyak_update(self.polyak, self.target_Q, self.Q)
        polyak_update(self.polyak, self.target_policy, self.policy)
