import torch
import numpy as np
from torch_scatter import scatter, scatter_max
from copy import copy
import math
from maro.simulator.scenarios.citi_bike.common import Action, DecisionEvent, DecisionType
import os
import shutil

def backup(source_pth, target_pth):
    '''
    if not os.path.exists(target_pth):
    # 如果目标路径不存在原文件夹的话就创建
        os.makedirs(target_pth)
    '''
    shutil.copytree(source_pth, target_pth)

def de_batchize(edges, graph_size):
    return edges % graph_size

def batch_split(data, dim=0):
    return np.split(data, indices_or_sections=data.shape[dim], axis=dim)

def batchize(batch_obs):
    graph_size = batch_obs[0]['node_cnt']
    batch_size = len(batch_obs)
    idx_inc = np.arange(batch_size)*graph_size

    acting_node_idx = np.array([e['acting_node_idx'] if 'acting_node_idx' in e else -1 for e in batch_obs]) + idx_inc
    actual_amount = np.vstack([e['actual_amount'] for e in batch_obs])
    action_edge_idx = np.hstack([batch_obs[i]['action_edge_idx']+idx_inc[i] for i in range(batch_size)])

    x = np.vstack([e['x'] for e in batch_obs])
    time = np.vstack([e['x'] for e in batch_obs])

    channel_cnt = len(batch_obs[0]['edge_idx_list'])
    edge_idx_list = [np.hstack([batch_obs[i]['edge_idx_list'][j]+idx_inc[i] for i in range(batch_size)]) for j in range(channel_cnt)]

    return {
        'acting_node_idx': acting_node_idx,
        'x': x,
        'time': time,
        'edge_idx_list': edge_idx_list,
        'action_edge_idx': action_edge_idx,
        'actual_amount': actual_amount,
        'node_cnt': graph_size,
    }

def batchize_exp(batch):
    if (not batch):
        return {}

    if isinstance(batch[0]['a'], tuple):
        a = np.hstack([e['a'][0] for e in batch])
    else:
        # a.shape: [2, action_cnt]
        a = np.hstack([e['a'] for e in batch])


    # state
    s = batchize([e['obs'] for e in batch])
    s_ = batchize([e['obs_'] for e in batch])
    tot_r = np.array([np.sum(e['r']) for e in batch])
    r = np.hstack([np.array(e['r']) for e in batch])

    gamma = np.hstack([np.array(e['gamma']) for e in batch])

    rlt = {
        'a': a,
        's': s,
        's_': s_,
        'r': r,
        'tot_r': tot_r,
        'gamma': gamma,
    }
    # supplement is handled by each algorithm (like GnnddPG), rather than outside.
    if 'supplement' in batch[0]:
        rlt['supplement'] = [e['supplement'] for e in batch]
    if 'self_r' in batch[0]:
        rlt['self_r'] = np.hstack([np.array(e['self_r']) for e in batch])
    return rlt



def from_numpy(dtype, device, *args):
    if not args:
        return None
    else:
        return [torch.from_numpy(x).type(dtype).to(device=device) for x in args]

def from_list(dtype, device, *args):
    return [dtype(x).to(device=device) for x in args]

def obs_to_torch(obs, device):
    x = from_numpy(torch.FloatTensor, device, obs['x'])[0]
    edge_idx_list = from_numpy(torch.LongTensor, device, *obs['edge_idx_list'])
    action_edge_idx = from_numpy(torch.LongTensor, device, obs['action_edge_idx'])[0]
    per_graph_size = obs['node_cnt']

    actual_amount = torch.FloatTensor(obs['actual_amount']).to(device=device)
    return x, edge_idx_list, action_edge_idx, actual_amount, per_graph_size

def time_obs_to_torch(obs, device):
    x = from_numpy(torch.FloatTensor, device, obs['x'])[0]
    time = from_numpy(torch.LongTensor, device, obs['time'])[0]
    edge_idx_list = from_numpy(torch.LongTensor, device, *obs['edge_idx_list'])
    action_edge_idx = from_numpy(torch.LongTensor, device, obs['action_edge_idx'])[0]
    per_graph_size = obs['node_cnt']

    actual_amount = torch.FloatTensor(obs['actual_amount']).to(device=device)
    return x, time, edge_idx_list, action_edge_idx, actual_amount, per_graph_size


def to_dense_adj(size, edge_index, edge_attr):
    rlt = edge_attr.new_zeros((*size, edge_attr.shape[1]), requires_grad=True)
    rlt[edge_index[0], edge_index[1]] = edge_attr
    return rlt

def polyak_update(polyak_factor, target_network, network):
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        target_param.data.copy_(polyak_factor*target_param.data + (1.0 - polyak_factor)*param.data)

def sparse_pooling(src, index=None, per_graph_size=None):
    if index is None:
        index = torch.arange(src.shape[0], dtype=torch.int64)/per_graph_size
        index = index.to(src.device)
    out, argmax = scatter_max(src, index, dim=0)
    return out

def compute_grad_norm(net, norm_type=2):
    normed_grads = [torch.norm(p.grad.detach(), norm_type) for p in net.parameters() if p.grad is not None]
    if len(normed_grads) == 0:
        return 0
    return torch.norm(torch.stack(normed_grads), norm_type)
