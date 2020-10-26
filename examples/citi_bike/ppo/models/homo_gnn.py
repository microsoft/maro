import math

import torch
from torch import nn, Tensor
from torch.distributions import Categorical
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, softmax
from torch_geometric.nn import GCNConv
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch_scatter import scatter_sum

from examples.citi_bike.ppo.models.transformer import CustomTransformerDecoderLayer
from examples.citi_bike.ppo.models.transformer import TransformerDecoder, TransformerEncoder, TransformerEncoderLayer
from examples.citi_bike.ppo.utils import to_dense_adj, sparse_pooling


device = torch.device('cuda:1')


class MultiChannelLinear(nn.Module):
    __constants__ = ['channels', 'in_features', 'out_features']
    in_features: int
    out_features: int
    channels: int
    weight: Tensor

    def __init__(self, channels: int, in_features: int, out_features: int, bias: bool = True) -> None:
        super(MultiChannelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.channels = channels
        self.weight = Parameter(torch.Tensor(channels, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(channels, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return torch.bmm(input, self.weight) + self.bias

    def extra_repr(self) -> str:
        return 'channels={}, in_features={}, out_features={}, bias={}'.format(
            1, self.in_features, self.out_features, self.bias is not None
        )


class EdgeConv(MessagePassing):
    def __init__(self, node_dim, edge_dim, out_dim):
        super(EdgeConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.layers = nn.Sequential(nn.Linear(node_dim + edge_dim, out_dim), nn.ReLU())

    def forward(self, x, edge_x, edge_index):
        # edge_index_0, edge_x_0 = remove_self_loops(edge_index, edge_attr=edge_x)
        # edge_index, edge_x = add_self_loops(edge_index, edge_weight=edge_x, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, edge_x=edge_x, norm=norm)

    def message(self, x_i, x_j, edge_x, norm):
        # x_j, edge_x : [E, dims*]
        feature = torch.cat((x_j, edge_x), dim=1)
        hidden = self.layers(feature)
        return norm.view(-1, 1) * hidden

    def update(self, aggr_out):
        return aggr_out


class Policy(nn.Module):
    def __init__(self, node_dim, neighbor_cnt, per_graph_size):
        super().__init__()
        self.node_dim = node_dim
        self.act_attention = nn.Sequential(torch.nn.Linear(node_dim * 2, 2), nn.Sigmoid())
        self.filter_func = nn.ReLU()
        self.per_graph_size = per_graph_size

    def forward(self, x, edge_index, actual_amount, noise_scale=0.0):
        # calculation attention
        # do not remove
        # edge_index, _ = remove_self_loops(edge_index)
        row, col = edge_index

        # att_threh = self.act_threshold(x[row])
        att_edge = self.act_attention(torch.cat((x[row], x[col]), dim=1))
        # act: [batch*per_graph_size, 2]
        # att = self.filter_func(att_edge - att_threh)
        att_edge = att_edge.reshape(-1, 21, 2)
        forgotten_idx = torch.topk(-att_edge, k=16, dim=1).indices
        att_edge = att_edge.scatter(1, forgotten_idx, 0)

        att_edge = att_edge.reshape(-1, 2)
        # att_sm = softmax(att_edge, row)
        batch_idx = row // self.per_graph_size
        # put a threshold 1.0
        batch_sum = scatter_sum(att_edge, index=batch_idx, dim=0)
        batch_sum[batch_sum < 1] = 1
        att_sm = att_edge / batch_sum[batch_idx]
        att_sm = torch.sum(att_sm * actual_amount, dim=1)
        return att_sm


class TwoStepPolicy(nn.Module):
    def __init__(self, node_dim, neighbor_cnt, per_graph_size):
        super().__init__()
        self.node_dim = node_dim
        self.choice_attention = nn.Sequential(torch.nn.Linear(node_dim * 2, 2), nn.ReLU())
        self.amount_attention = nn.Sequential(torch.nn.Linear(node_dim * 2, 2), nn.ReLU())
        self.filter_func = nn.ReLU()
        self.per_graph_size = per_graph_size

    def forward(self, x, edge_index, actual_amount, noise_scale=0.0):
        # calculation attention
        row, col = edge_index
        # att_threh = self.act_threshold(x[row])
        att_edge = self.choice_attention(torch.cat((x[row], x[col]), dim=1))
        amount_edge = self.amount_attention(torch.cat((x[row], x[col]), dim=1))
        # att_edge: [edge_cnt, 2]

        att_sm = softmax(att_edge, row)
        att_sm[actual_amount == 0] = 0
        att = torch.sum(att_sm, dim=1)

        amount = amount_edge * actual_amount
        amount = torch.sum(amount_edge, dim=-1)
        return att, amount


class AttAmtPolicy(nn.Module):
    def __init__(self, node_dim, neighbor_cnt, per_graph_size):
        super().__init__()
        self.amt_bucket = 30
        self.node_dim = node_dim
        self.mlp_att = nn.Sequential(torch.nn.Linear(node_dim * (neighbor_cnt + 1), 256), nn.ReLU(inplace=False),
                                     torch.nn.Linear(256, 128), nn.ReLU(inplace=False),
                                     torch.nn.Linear(128, neighbor_cnt), nn.Softmax(-1))
        self.mlp_amt = nn.Sequential(torch.nn.Linear(node_dim * (neighbor_cnt + 1) + 1, 256), nn.ReLU(inplace=False),
                                     torch.nn.Linear(256, 128), nn.ReLU(inplace=False),
                                     torch.nn.Linear(128, self.amt_bucket), nn.Softmax(-1))
        self.critic_layer = nn.Sequential(torch.nn.Linear(node_dim * (neighbor_cnt + 1) + 1, 512),
                                          nn.ReLU(inplace=False), torch.nn.Linear(512, 512),
                                          nn.ReLU(inplace=False), torch.nn.Linear(512, 449))
        self.filter_func = nn.ReLU()
        self.softmax = nn.Softmax(-1)
        self.per_graph_size = per_graph_size
        self.neighbor_cnt = neighbor_cnt

    def forward(self, x, edge_index, actual_amount, noise_scale=0.0):
        # calculation attention
        row, col = edge_index
        # att_threh = self.act_threshold(x[row])
        # feature_idx = torch.cat((row[::self.neighbor_cnt],col),-1)
        feature_idx = col
        attfeature = x[feature_idx].reshape(-1, 336)
        att = self.mlp_att(attfeature)
        m = Categorical(att)
        choice = m.sample()
        if(actual_amount[0, 0] > 0):
            masked_value = actual_amount[choice, 0]
            sign = 1
        else:
            masked_value = - actual_amount[choice, 1]
            sign = -1
        # prepare the mask
        masked_idx = torch.tensor(masked_value).reshape(-1, 1)
        masked_arr = torch.zeros((masked_idx.shape[0], self.amt_bucket)).to(device)
        for i in range(masked_idx.shape[0]):
            temp = int(math.ceil(masked_idx[i].item() / 0.1 / 2))
            temp = min(30, temp)
            temp = max(1, temp)
            masked_arr[i, temp:] = float('-inf')

        actual_value = masked_value.reshape(-1, 1)
        amtfeature = torch.cat((attfeature, actual_value), -1)
        amount = self.mlp_amt(amtfeature)
        amount = self.softmax(amount + masked_arr)
        m_ = Categorical(amount)
        cnt = m_.sample()

        # change the choice
        # amtfeature = attfeature
        return choice, cnt * sign, att, amount

    def evaluate(self, x, edge_index, actual_amount, real_act):
        # calculation attention
        row, col = edge_index
        # att_threh = self.act_threshold(x[row])
        # row_compact = row[::self.neighbor_cnt].reshape(-1,1)
        feature_idx = col
        # feature_idx = torch.cat((row_compact,col.reshape(-1,self.neighbor_cnt)),-1).reshape(-1)
        attfeature = x[feature_idx].reshape(-1, 336)
        att = self.mlp_att(attfeature)
        # att[(actual_amount.sum(-1)==0)] = 0
        actual_amount = actual_amount.reshape(-1, self.neighbor_cnt + 1, 2)
        masked_value = torch.zeros((actual_amount.shape[0])).to(device)
        for i in range(actual_amount.shape[0]):
            if(actual_amount[i, 0, 0] > 0):
                masked_value[i] = actual_amount[i, real_act[i], 0]
            else:
                masked_value[i] = - actual_amount[i, real_act[i], 1]
        # prepare the mask
        masked_idx = torch.tensor(masked_value).reshape(-1, 1)
        masked_arr = torch.zeros((masked_idx.shape[0], self.amt_bucket)).to(device)
        for i in range(masked_idx.shape[0]):
            temp = int(math.ceil(masked_idx[i].item() / 0.1 / 2))
            temp = min(30, temp)
            temp = max(1, temp)
            masked_arr[i, temp:] = float('-inf')
        actual_value = masked_value.reshape(-1, 1)
        amtfeature = torch.cat((attfeature, actual_value), -1)
        amount = self.mlp_amt(amtfeature)
        amount = self.softmax(amount + masked_arr)
        m_ = Categorical(amount)
        cnt = m_.sample()
        return cnt, att, amount

    def value(self, x, edge_index, actual_amount, real_act=None):
        row, col = edge_index

        attfeature = x[col].reshape(-1, 336)
        actual_amount = actual_amount.reshape(-1, self.neighbor_cnt + 1, 2)
        if(real_act is not None):
            masked_value = torch.zeros((actual_amount.shape[0])).to(device)
            for i in range(actual_amount.shape[0]):
                if(actual_amount[i, 0, 0] > 0):
                    masked_value[i] = actual_amount[i, real_act[i], 0]
                else:
                    masked_value[i] = - actual_amount[i, real_act[i], 1]
            actual_value = masked_value.reshape(-1, 1)
        else:
            actual_value = torch.zeros((actual_amount.shape[0], 1)).to(device)
        amtfeature = torch.cat((attfeature, actual_value), -1)
        state_value = self.critic_layer(amtfeature)
        return state_value


class AmtPolicy(nn.Module):
    def __init__(self, node_dim, neighbor_cnt, per_graph_size):
        super().__init__()
        self.amt_bucket = 30
        self.node_dim = node_dim
        self.mlp_amt = nn.Sequential(torch.nn.Linear(node_dim * (neighbor_cnt + 1) + 1, 256), nn.ReLU(inplace=False),
                                     torch.nn.Linear(256, 128), nn.ReLU(inplace=False),
                                     torch.nn.Linear(128, self.amt_bucket), nn.Softmax(-1))
        self.critic_layer = nn.Sequential(torch.nn.Linear(node_dim * (neighbor_cnt + 1) + 1, 512),
                                          nn.ReLU(inplace=False), torch.nn.Linear(512, 512),
                                          nn.ReLU(inplace=False), torch.nn.Linear(512, 449))
        self.filter_func = nn.ReLU()
        self.softmax = nn.Softmax(-1)
        self.per_graph_size = per_graph_size
        self.neighbor_cnt = neighbor_cnt

    def forward(self, x, edge_index, actual_amount, noise_scale=0.0):
        # calculation attention
        row, col = edge_index
        attfeature = x[col].reshape(-1, 336)
        choice = torch.tensor([0])
        if(actual_amount[0, 0] > 0):
            masked_value = actual_amount[0, 0]
            sign = 1
        else:
            masked_value = - actual_amount[0, 1]
            sign = -1
        # prepare the mask
        masked_idx = torch.tensor(masked_value).reshape(-1, 1)
        masked_arr = torch.zeros((masked_idx.shape[0], self.amt_bucket)).to(device)
        for i in range(masked_idx.shape[0]):
            temp = int(math.ceil(masked_idx[i].item() / 0.1 / 2))
            temp = min(30, temp)
            temp = max(1, temp)
            masked_arr[i, temp:] = float('-inf')

        actual_value = masked_value.reshape(-1, 1)
        amtfeature = torch.cat((attfeature, actual_value), -1)
        amount = self.mlp_amt(amtfeature)
        amount = self.softmax(amount + masked_arr)
        m_ = Categorical(amount)
        cnt = m_.sample()

        return choice, cnt * sign, amount

    def evaluate(self, x, edge_index, actual_amount):
        # calculation attention
        row, col = edge_index
        attfeature = x[col].reshape(-1, 336)
        actual_amount = actual_amount.reshape(-1, self.neighbor_cnt + 1, 2)
        masked_value = torch.zeros((actual_amount.shape[0])).to(device)
        for i in range(actual_amount.shape[0]):
            if(actual_amount[i, 0, 0] > 0):
                masked_value[i] = actual_amount[i, 0, 0]
            else:
                masked_value[i] = - actual_amount[i, 0, 1]
        # prepare the mask
        masked_idx = torch.tensor(masked_value).reshape(-1, 1)
        masked_arr = torch.zeros((masked_idx.shape[0], self.amt_bucket)).to(device)
        for i in range(masked_idx.shape[0]):
            temp = int(math.ceil(masked_idx[i].item() / 0.1 / 2))
            temp = min(30, temp)
            temp = max(1, temp)
            masked_arr[i, temp:] = float('-inf')
        actual_value = masked_value.reshape(-1, 1)
        amtfeature = torch.cat((attfeature, actual_value), -1)
        amount = self.mlp_amt(amtfeature)
        amount = self.softmax(amount + masked_arr)
        return amount

    def value(self, x, edge_index, actual_amount):
        row, col = edge_index

        attfeature = x[col].reshape(-1, 336)
        actual_amount = actual_amount.reshape(-1, self.neighbor_cnt + 1, 2)
        masked_value = torch.zeros((actual_amount.shape[0])).to(device)
        for i in range(actual_amount.shape[0]):
            if(actual_amount[i, 0, 0] > 0):
                masked_value[i] = actual_amount[i, 0, 0]
            else:
                masked_value[i] = - actual_amount[i, 0, 1]
        actual_value = masked_value.reshape(-1, 1)
        amtfeature = torch.cat((attfeature, actual_value), -1)
        state_value = self.critic_layer(amtfeature)
        return state_value


class AttPolicy(nn.Module):
    def __init__(self, node_dim, neighbor_cnt, per_graph_size):
        super().__init__()
        self.amt_bucket = 30
        self.node_dim = node_dim
        self.mlp_att = nn.Sequential(torch.nn.Linear(node_dim * (neighbor_cnt + 1), 256), nn.ReLU(inplace=False),
                                     torch.nn.Linear(256, 128), nn.ReLU(inplace=False),
                                     torch.nn.Linear(128, neighbor_cnt), nn.Softmax(-1))
        self.critic_layer = nn.Sequential(torch.nn.Linear(node_dim * (neighbor_cnt + 1) + 1, 512),
                                          nn.ReLU(inplace=False), torch.nn.Linear(512, 512),
                                          nn.ReLU(inplace=False), torch.nn.Linear(512, 449))
        self.filter_func = nn.ReLU()
        self.softmax = nn.Softmax(-1)
        self.per_graph_size = per_graph_size
        self.neighbor_cnt = neighbor_cnt
        self.perc = 0.5

    def forward(self, x, edge_index, actual_amount, noise_scale=0.0):
        # calculation attention
        row, col = edge_index
        # att_threh = self.act_threshold(x[row])
        # feature_idx = torch.cat((row[::self.neighbor_cnt],col),-1)
        feature_idx = col
        attfeature = x[feature_idx].reshape(-1, 336)
        att = self.mlp_att(attfeature)
        m = Categorical(att)
        choice = m.sample()
        cnt = 0.5 * actual_amount[choice, 0]
        return choice, cnt, att

    def evaluate(self, x, edge_index, actual_amount, real_act):
        # calculation attention
        row, col = edge_index
        # att_threh = self.act_threshold(x[row])
        # row_compact = row[::self.neighbor_cnt].reshape(-1,1)
        feature_idx = col
        # feature_idx = torch.cat((row_compact,col.reshape(-1,self.neighbor_cnt)),-1).reshape(-1)
        attfeature = x[feature_idx].reshape(-1, 336)
        att = self.mlp_att(attfeature)
        # att[(actual_amount.sum(-1)==0)] = 0
        actual_amount = actual_amount.reshape(-1, self.neighbor_cnt + 1, 2)
        masked_value = torch.zeros((actual_amount.shape[0])).to(device)
        for i in range(actual_amount.shape[0]):
            if(actual_amount[i, 0, 0] > 0):
                masked_value[i] = actual_amount[i, real_act[i], 0]
            else:
                masked_value[i] = - actual_amount[i, real_act[i], 1]
        # prepare the mask
        masked_idx = torch.tensor(masked_value).reshape(-1, 1)
        masked_arr = torch.zeros((masked_idx.shape[0], self.amt_bucket)).to(device)
        for i in range(masked_idx.shape[0]):
            temp = int(math.ceil(masked_idx[i].item() / 0.1 / 2))
            temp = min(30, temp)
            temp = max(1, temp)
            masked_arr[i, temp:] = float('-inf')
        actual_value = masked_value.reshape(-1, 1)
        amtfeature = torch.cat((attfeature, actual_value), -1)
        amount = self.mlp_amt(amtfeature)
        amount = self.softmax(amount + masked_arr)
        m_ = Categorical(amount)
        cnt = m_.sample()
        return cnt, att, amount

    def value(self, x, edge_index, actual_amount, real_act=None):
        row, col = edge_index

        attfeature = x[col].reshape(-1, 336)
        actual_amount = actual_amount.reshape(-1, self.neighbor_cnt + 1, 2)
        if(real_act is not None):
            masked_value = torch.zeros((actual_amount.shape[0])).to(device)
            for i in range(actual_amount.shape[0]):
                if(actual_amount[i, 0, 0] > 0):
                    masked_value[i] = actual_amount[i, real_act[i], 0]
                else:
                    masked_value[i] = - actual_amount[i, real_act[i], 1]
            actual_value = masked_value.reshape(-1, 1)
        else:
            actual_value = torch.zeros((actual_amount.shape[0], 1)).to(device)
        amtfeature = torch.cat((attfeature, actual_value), -1)
        state_value = self.critic_layer(amtfeature)
        return state_value


class DestSelectionPolicy(nn.Module):
    '''
    this only select one destination
    '''
    def __init__(self, node_dim, neighbor_cnt, per_graph_size):
        super().__init__()
        self.node_dim = node_dim
        self.choice_attention = nn.Sequential(torch.nn.Linear(node_dim * 2, 2), nn.ReLU())
        self.per_graph_size = per_graph_size

    def forward(self, x, edge_index, actual_amount, noise_scale=0.0):
        # calculation attention
        row, col = edge_index

        # att_threh = self.act_threshold(x[row])
        att_edge = self.choice_attention(torch.cat((x[row], x[col]), dim=1))
        # att_edge: [edge_cnt, 2]

        att_sm = softmax(att_edge, row)
        att_sm[actual_amount == 0] = 0

        att = torch.sum(att_sm, dim=1)
        return att


class SparsePolicy(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.node_dim = node_dim
        hidden_edge_dim = 4
        self.edge_emb = torch.nn.Linear(edge_dim, hidden_edge_dim)
        self.act_attention = nn.Sequential(torch.nn.Linear(node_dim * 2 + hidden_edge_dim, 2), nn.Sigmoid())
        self.act_threshold = nn.Sequential(torch.nn.Linear(node_dim, 2), nn.Sigmoid())
        self.filter_func = nn.ReLU()

    def forward(self, x, edge_x, edge_index, actual_amount, per_graph_size):
        # calculation attention
        row, col = edge_index
        hid_edge_x = self.edge_emb(edge_x)
        att_threh = self.act_threshold(x[row])
        att_edge = self.act_attention(torch.cat((x[row], x[col], hid_edge_x), dim=1))
        # act: [batch*per_graph_size, 2]
        att = self.filter_func(att_edge - att_threh)
        # att_edge = att_edge.reshape(-1, 21, 2)
        # forgotten_idx = torch.topk(-att_edge, k=15, dim=1).indices
        # att_edge = att_edge.scatter(1, forgotten_idx, 0)
        # potentially wrong, check if need 'row/per_graph_size'
        # att_edge = att_edge.reshape(-1, 2)
        # att_sm = softmax(att_edge, row)
        batch_idx = row // per_graph_size
        batch_sum = scatter_sum(att, index=batch_idx, dim=0) + 0.00001
        att_sm = att / batch_sum[batch_idx]
        att_sm = torch.sum(att_sm * actual_amount, dim=1)
        return att_sm


class Reslayer(nn.Module):
    def __init__(self, in_dim, activation_func=nn.ReLU):
        super().__init__()
        self.layer = nn.Linear(in_dim, in_dim)
        self.act = activation_func()

    def forward(self, x):
        return self.act(self.layer(x) + x)


class QHeader(nn.Module):
    def __init__(self, in_dim, res_cnt, activation_func=nn.ReLU):
        super().__init__()
        res_layers = [Reslayer(in_dim, activation_func=activation_func) for _ in range(res_cnt)]
        self.q_header = nn.Sequential(*res_layers, nn.Linear(in_dim, 1), activation_func())

    def forward(self, x):
        return self.q_header(x)


'''
class GNNBackend(nn.Module):
    def __init__(self, node_dim, out_dim, channel_cnt=1):
        super().__init__()
        self.node_dim = node_dim
        self.channels = nn.ModuleList([GCNConv(node_dim, out_dim) for _ in range(channel_cnt)])
        self.output_emb = nn.Sequential(nn.Linear(out_dim*channel_cnt, out_dim), nn.ReLU())

    def forward(self, x, edge_idx_list):
        ''
        this can only handle the graph with the same size in the graph
        per_graph_size: the size of each graph
        ''
        hid_list = [conv(x, edge_idx) for conv, edge_idx in zip(self.channels, edge_idx_list)]
        hid = torch.cat(hid_list, axis=0)
        x = self.output_emb(hid)
        return x
'''


class STGNNBackend(nn.Module):
    def __init__(self, node_dim, out_dim, channel_cnt=1):
        super().__init__()
        self.node_dim = node_dim
        # self.temporal_rnn = nn.GRU(input_size=self.node_dim, hidden_size=self.temporal_size, num_layers=2,
        #                            batch_first=True)
        self.temporal_size = 32
        self.spatial_size = 32
        self.fwd = nn.Sequential(nn.Linear(self.node_dim, self.temporal_size), nn.ReLU(),
                                 nn.Linear(self.temporal_size, self.temporal_size), nn.ReLU())
        self.spatial_emb1 = GCNConv(self.temporal_size, self.temporal_size)
        self.spatial_emb2 = GCNConv(self.temporal_size, self.temporal_size)
        # self.output_emb = nn.Sequential(nn.Linear(self.spatial_size*channel_cnt, out_dim), nn.ReLU())
        self.output_emb = nn.Sequential(nn.Linear(self.temporal_size, out_dim), nn.ReLU())

    def forward(self, x, edge):
        cur_x = x[-1]
        tx = self.fwd(cur_x)
        stx1 = self.spatial_emb1(tx, edge)
        stx1 = F.relu(stx1 + tx)
        stx2 = self.spatial_emb2(stx1, edge)
        output = self.output_emb(stx2)
        return output


class LinearBackend(nn.Module):
    def __init__(self, node_dim, out_dim, channel_cnt=1):
        super().__init__()
        self.node_dim = node_dim
        self.temporal_size = 32
        self.spatial_size = 32
        self.fwd = nn.Sequential(nn.Linear(self.node_dim, self.temporal_size), nn.ReLU(),
                                 nn.Linear(self.temporal_size, self.temporal_size), nn.ReLU(),
                                 nn.Linear(self.temporal_size, out_dim), nn.ReLU())

    def forward(self, x, edge_idx_list):
        # x.shape batch_cnt * sequence_len * node_dim
        cur_x = x[:, 0]
        output = self.fwd(cur_x)
        return output


class LinearBackendtpsc(nn.Module):
    def __init__(self, node_dim, out_dim, channel_cnt=1):
        super().__init__()
        self.node_dim = node_dim
        self.temporal_size = 32
        self.spatial_size = 32
        self.fwd = nn.Sequential(nn.Linear(self.node_dim, self.temporal_size), nn.ReLU(),
                                 nn.Linear(self.temporal_size, self.temporal_size), nn.ReLU(),
                                 nn.Linear(self.temporal_size, out_dim), nn.ReLU())

    def forward(self, x, edge_idx_list):
        # x.shape batch_cnt*sequence_len*node_dim
        cur_x = x[-1]
        output = self.fwd(cur_x)
        return output


class GNNBackend(nn.Module):
    def __init__(self, time_window, node_dim, output_dim):
        '''
        month = np.zeros(tot_time_cnt, dtype=np.int)
        season = np.zeros(tot_time_cnt, dtype=np.int)
        dayofweek = np.zeros(tot_time_cnt, dtype=np.int)
        dayofmonth = np.zeros(tot_time_cnt, dtype=np.int)
        hourofday = np.zeros(tot_time_cnt, dtype=np.int)
        minuteofhour = np.zeros(tot_time_cnt, dtype=np.int)
        '''
        super().__init__()
        self.month_emb = nn.Embedding(num_embeddings=12, embedding_dim=3)
        self.season_emb = nn.Embedding(num_embeddings=4, embedding_dim=2)
        self.dayofweek_emb = nn.Embedding(num_embeddings=7, embedding_dim=2)
        self.dayofmonth_emb = nn.Embedding(num_embeddings=31, embedding_dim=4)
        self.hourofday = nn.Embedding(num_embeddings=24, embedding_dim=4)
        self.minuteofhour = nn.Embedding(num_embeddings=3, embedding_dim=2)
        idim = node_dim + (3 + 2 + 2 + 4 + 4 + 2)
        self.bn = nn.BatchNorm1d(node_dim)
        thid = 16
        self.temporal_emb = nn.GRU(input_size=idim, hidden_size=thid)
        shid = thid * time_window
        self.output_dim = output_dim
        hhid = 32
        self.final_header = nn.Sequential(nn.Linear(shid, hhid), nn.ReLU(), nn.Linear(hhid, output_dim))

    def forward(self, x, edge, time):
        '''
        x.shape: (S, B, Dim)
        time[*].shape: (S, B)
        edge.shape: (2, edge_cnt)
        '''
        seq, batch, _ = x.shape
        # x = self.bn(x.reshape(seq*batch, -1)).reshape(seq, batch, -1)
        s, m, dw, dm, hd, mh = torch.split(time, 1, dim=-1)
        s = self.season_emb(torch.squeeze(s, -1))
        m = self.month_emb(torch.squeeze(m, -1))
        dw = self.dayofweek_emb(torch.squeeze(dw, -1))
        dm = self.dayofmonth_emb(torch.squeeze(dm, -1))
        hd = self.hourofday(torch.squeeze(hd, -1))
        mh = self.minuteofhour(torch.squeeze(mh, -1))
        x = torch.cat((x, s, m, dw, dm, hd, mh), dim=-1)

        tx, _ = self.temporal_emb(x)
        tx = torch.transpose(tx, 1, 0).reshape(batch, -1)
        output = self.final_header(tx)
        return output


class SimpleBackend(nn.Module):
    def __init__(self, node_dim, output_dim, time_window=10):
        super().__init__()
        self.node_dim, self.out_dim, self.time_steps = node_dim, output_dim, time_window
        self.input_dim = node_dim * time_window
        hidden_dim = 64
        self.emb = nn.Sequential(nn.Linear(self.node_dim * time_window, hidden_dim), nn.BatchNorm1d(hidden_dim),
                                 nn.ReLU(), nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU())

    def forward(self, x, edge_list, time):
        '''
        x.shape: (S, B, Dim)
        time[*].shape: (S, B)
        edge.shape: (2, edge_cnt)
        '''
        seq_len, batch, dim = x.shape
        assert(seq_len == self.time_steps and dim == self.node_dim)
        x = torch.transpose(x, 1, 0)
        output = self.emb(x.reshape(-1, self.input_dim)).reshape(batch, self.out_dim)
        return output


class STTransBackend(nn.Module):
    def __init__(self, node_dim, out_dim, channel_cnt=1):
        super().__init__()
        self.node_dim = node_dim
        self.encoderLayer = TransformerEncoderLayer(d_model=self.node_dim, nhead=2, dropout=0)
        self.transformerencoder = TransformerEncoder(self.encoderLayer, 1)
        self.decoderLayer = CustomTransformerDecoderLayer(d_model=self.node_dim, nhead=2, dropout=0)
        self.transformerdecoder = TransformerDecoder(self.decoderLayer, 1)
        self.output_emb = nn.Sequential(nn.Linear(self.node_dim, out_dim), nn.ReLU())

    def forward(self, x, edge_idx_list):
        # x.shape batch_cnt*sequence_len*node_dim
        x = torch.transpose(x, 0, 1)
        memory = self.transformerencoder(x)
        x_cur = x[-1].unsqueeze(0)
        tgt = self.transformerdecoder(x_cur, memory)
        output = self.output_emb(tgt).squeeze(0)
        return output


class Q(nn.Module):
    def __init__(self, node_dim, per_graph_size, res_cnt=2, activation_func=nn.ReLU):
        super().__init__()
        action_hidden = 4
        self.action_emb = nn.Sequential(nn.Linear(1, action_hidden), nn.ReLU())
        res_layers = [Reslayer(node_dim + action_hidden, activation_func=activation_func) for _ in range(res_cnt)]
        self.q_header = nn.Sequential(*res_layers, nn.Linear(node_dim + action_hidden, 1), activation_func())
        self.per_graph_size = per_graph_size

    def forward(self, x, actions, action_edge_index):
        '''
        orgainize edge features
        '''
        row, col = action_edge_index
        row = row / self.per_graph_size
        col = col % self.per_graph_size
        batch_size = row.max().item() + 1
        batch_edge_index = torch.stack((row, col), dim=0)
        # batch_actions: [batch*per_graph_size, 1]
        batch_actions = to_dense_adj((batch_size, self.per_graph_size), batch_edge_index,
                                     edge_attr=actions.reshape(-1, 1)).view(-1, 1)
        # batch_action_edges: [batch*per_graph_size, edge_dim]
        batch_action_feat = self.action_emb(batch_actions)

        '''
        get Q value
        '''
        # batch_q: [batch*per_graph_size, dim]
        q_feat = torch.cat((x, batch_action_feat), dim=1)
        Qs = self.q_header(q_feat)
        return Qs


class TwoHeadQ(nn.Module):
    def __init__(self, node_dim, per_graph_size, res_cnt=2, activation_func=nn.ReLU):
        super().__init__()
        action_hidden = 4
        self.action_emb = nn.Sequential(nn.Linear(2, action_hidden), nn.ReLU())
        res_layers = [Reslayer(node_dim + action_hidden, activation_func=activation_func) for _ in range(res_cnt)]
        self.q_header = nn.Sequential(*res_layers, nn.Linear(node_dim + action_hidden, 1))

        res_layers = [Reslayer(node_dim + action_hidden, activation_func=activation_func) for _ in range(res_cnt)]
        self.tot_q_header = nn.Sequential(*res_layers, nn.Linear(node_dim + action_hidden, 1))
        self.per_graph_size = per_graph_size

    def forward(self, x, actions, action_edge_index):
        '''
        orgainize edge features
        '''
        row, col = action_edge_index
        row = row / self.per_graph_size
        col = col % self.per_graph_size
        batch_size = row.max().item() + 1
        batch_edge_index = torch.stack((row, col), dim=0)
        # batch_actions: [batch*per_graph_size, 1]
        batch_actions = to_dense_adj((batch_size, self.per_graph_size), batch_edge_index,
                                     edge_attr=actions).view(-1, 2)
        # batch_action_edges: [batch*per_graph_size, edge_dim]
        batch_action_feat = self.action_emb(batch_actions)

        '''
        get Q value
        '''
        # batch_q: [batch*per_graph_size, dim]
        q_feat = torch.cat((x, batch_action_feat), dim=1)
        Qs = self.q_header(q_feat)

        q_pooling = sparse_pooling(q_feat, per_graph_size=self.per_graph_size)
        Q = self.tot_q_header(q_pooling)
        return Qs.reshape(-1), Q.reshape(-1)


class TwoHeadLocalQ(nn.Module):
    def __init__(self, node_dim, per_graph_size, local_graph_size, res_cnt=2, activation_func=nn.ReLU):
        super().__init__()
        action_hidden = 4
        self.action_emb = nn.Sequential(nn.Linear(2, action_hidden), nn.ReLU())
        res_layers = [Reslayer(node_dim + action_hidden, activation_func=activation_func) for _ in range(res_cnt)]
        self.q_header = nn.Sequential(*res_layers, nn.Linear(node_dim + action_hidden, 1))

        self.tot_dim = (node_dim + action_hidden) * local_graph_size
        res_layers = [Reslayer(self.tot_dim, activation_func=activation_func) for _ in range(res_cnt)]
        self.tot_q_header = nn.Sequential(*res_layers, nn.Linear(self.tot_dim, 1))
        self.per_graph_size = per_graph_size
        self.local_graph_size = torch.FloatTensor(local_graph_size)

    def forward(self, x, actions, action_edge_idx_list):
        row, col = action_edge_idx_list
        x_idx = col.reshape(-1, 20)
        x_idx = x_idx.reshape(-1)
        action_feat = self.action_emb(actions)
        neighbor_feat = torch.cat((x[x_idx], action_feat), dim=1)
        Qs = self.q_header(neighbor_feat)

        tot_q_feat = neighbor_feat.reshape(-1, self.tot_dim)
        Q = self.tot_q_header(tot_q_feat)
        return Qs.reshape(-1), Q.reshape(-1)
