import torch
from torch import nn
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, degree, softmax
from examples.citi_bike.ppo.utils import to_dense_adj, sparse_pooling
from examples.citi_bike.ppo.models.transformer import TransformerDecoder,TransformerEncoder,TransformerEncoderLayer,CustomTransformerDecoderLayer
from examples.citi_bike.ppo.models.homo_gnn import MultiChannelLinear
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_sum, scatter_mean
from torch.distributions import Categorical
import math
import numpy as np

class AttTransPolicy(nn.Module):
    def __init__(self, node_dim, neighbor_cnt, per_graph_size):
        super().__init__()
        self.node_dim = node_dim
        self.softmax = nn.Softmax(-1)
        self.per_graph_size = per_graph_size
        self.neighbor_cnt = neighbor_cnt
        self.amt_resolution = 11
        self.amt_step = 1.0/(self.amt_resolution-1)
        self.encoderLayer = TransformerEncoderLayer(d_model=self.node_dim, nhead=2, dropout=0.0)
        self.transformerencoder = TransformerEncoder(self.encoderLayer, 1)
        self.decoderLayer = CustomTransformerDecoderLayer(d_model=self.node_dim, nhead=2, dropout=0.0)
        self.transformerdecoder = TransformerDecoder(self.decoderLayer,2)

        self.amt_hidden = 32
        self.amt_encoder_layer = TransformerEncoderLayer(d_model=self.node_dim, nhead=2, dropout=0.0)
        self.amt_header = nn.Sequential(nn.Linear(self.node_dim*2, self.amt_hidden), nn.ReLU(), nn.Linear(self.amt_hidden, self.amt_resolution))
        self.amt_softmax = nn.Softmax(-1)

        '''
        critic_encoder_layer = TransformerEncoderLayer(d_model=self.node_dim, nhead=2)
        self.critic_transformer = TransformerEncoder(critic_encoder_layer, 1)
        self.critic_header = nn.Sequential(nn.Linear(self.node_dim, 16), nn.ReLU(), nn.Linear(16, 1))
        '''

        self.critic_hidden_dim = 16
        # self.critic_headers = nn.Sequential(MultiChannelLinear(self.per_graph_size, self.node_dim, self.critic_hidden_dim), nn.ReLU(), MultiChannelLinear(self.per_graph_size, self.critic_hidden_dim, 1))
        self.critic_headers = nn.Sequential(MultiChannelLinear(self.per_graph_size, self.node_dim, self.critic_hidden_dim), nn.ReLU(), MultiChannelLinear(self.per_graph_size, self.critic_hidden_dim, 1))

    def determine_amount(self, x, actual_amount, src, dest_idx, sample=True):
        # chosen_dest.shape: B*D
        actual_amount = actual_amount.reshape(-1, self.neighbor_cnt)[:, 0]
        batch_size = actual_amount.shape[0]
        sign = actual_amount.new_ones((batch_size, 1), dtype=torch.int, requires_grad=False)
        sign[actual_amount < 0, 0] = -1
        batch_idx = torch.arange(batch_size, requires_grad=False).to(x.device)
        src_x = x[src].reshape(1, batch_size, self.node_dim)
        x = x.reshape(batch_size, self.per_graph_size, self.node_dim)
        dest_x = x[batch_idx, dest_idx].reshape(1, batch_size, self.node_dim)

        seq_x = torch.cat((src_x, dest_x), axis=0)
        emb_x = self.amt_encoder_layer(seq_x)
        emb_x = torch.transpose(emb_x, 1, 0).reshape(batch_size, self.node_dim*2)
        emb_x = self.amt_header(emb_x)
        att = self.amt_softmax(emb_x)

        amt_choice = None
        if sample:
            amt_m = Categorical(att)
            amt_choice = amt_m.sample()

        return amt_choice, att

    def choose_destination(self, x, edge_index, actual_amount, acting_node, noise_scale=0.0, sample=True):
        # calculation attention
        row, col = edge_index

        # print("col",col)
        # get the source group
        actual_amount = actual_amount.reshape(-1, self.neighbor_cnt)[:, 0]
        batch_size = actual_amount.shape[0]
        sign = actual_amount.new_ones((batch_size, 1), dtype=torch.int, requires_grad=False)
        sign[actual_amount < 0, 0] = -1

        ensrc = x[col].reshape(batch_size, self.neighbor_cnt, self.node_dim)
        ensrc = torch.transpose(ensrc, 0, 1)
        desrc = x[acting_node].reshape(1,batch_size,self.node_dim)
        memory = self.transformerencoder(ensrc)
        memory = memory * sign
        tgt = self.transformerdecoder(desrc,memory)
        memory1 = torch.transpose(memory, 0, 1)
        memory2 = torch.transpose(memory1, 1, 2)

        tgt_temp = torch.transpose(tgt,0,1)
        att = torch.bmm(tgt_temp, memory2)
        att = self.softmax(att.squeeze(1))

        choice = None
        if(sample):
            m = Categorical(att)
            choice = m.sample()

        # cnt = self.perc * actual_amount[batch_idx, choice]
        return choice, att

    def value(self, x, *args):
        '''
        x = torch.transpose(x.reshape(-1, self.per_graph_size, self.node_dim), 0, 1)
        output = self.critic_headers(x).reshape(self.per_graph_size, -1)
        return torch.transpose(output, 0, 1)
        '''
        x = torch.transpose(x.reshape(-1, self.per_graph_size, self.node_dim), 0, 1)
        output = self.critic_headers(x).reshape(self.per_graph_size, -1)
        return torch.transpose(output, 0, 1)
