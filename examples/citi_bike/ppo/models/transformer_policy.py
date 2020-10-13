import torch
from torch import nn
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, degree, softmax
from examples.citi_bike.enc_gat.utils import to_dense_adj, sparse_pooling
from examples.citi_bike.enc_gat.models.transformer import TransformerDecoder,TransformerEncoder,TransformerEncoderLayer,CustomTransformerDecoderLayer
from examples.citi_bike.enc_gat.models.homo_gnn import MultiChannelLinear
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
        self.perc = 0.5
        self.encoderLayer = TransformerEncoderLayer(d_model=self.node_dim, nhead=2,dropout=0)
        self.transformerencoder = TransformerEncoder(self.encoderLayer, 1)
        self.decoderLayer = CustomTransformerDecoderLayer(d_model=self.node_dim, nhead=2,dropout=0)
        self.transformerdecoder = TransformerDecoder(self.decoderLayer,1)

        '''
        critic_encoder_layer = TransformerEncoderLayer(d_model=self.node_dim, nhead=2)
        self.critic_transformer = TransformerEncoder(critic_encoder_layer, 1)
        self.critic_header = nn.Sequential(nn.Linear(self.node_dim, 16), nn.ReLU(), nn.Linear(16, 1))
        '''

        self.critic_hidden_dim = 16
        self.critic_transformer_layer = TransformerEncoderLayer(d_model=self.node_dim, nhead=2)
        self.critic_encoder = TransformerEncoder(self.encoderLayer, 1)
        self.critic_mlp = nn.Sequential(nn.Linear(self.node_dim*2, 16), nn.ReLU(),
                                        nn.Linear(16,16),nn.ReLU(),
                                        nn.Linear(16, 1))
    def forward(self, x, edge_index, actual_amount, noise_scale=0.0):
        # calculation attention
        row, col = edge_index
        
        # print("col",col)
        # get the source group
        actual_amount = torch.sum(actual_amount.reshape(-1, self.neighbor_cnt+1, 2), axis=-1)
        batch_size = actual_amount.shape[0]
        sign = actual_amount.new_ones((batch_size, 1), dtype=torch.int, requires_grad=False)
        sign[actual_amount[:, 0] < 0, 0] = -1

        desrc_idx = col.reshape(-1,self.neighbor_cnt+1)[:,-1].reshape(-1)
        # ensrc_idx = col.reshape(-1, self.neighbor_cnt+1)[:, :-1].reshape(-1)
        ensrc = x[col].reshape(-1,self.neighbor_cnt+1, self.node_dim)
        ensrc = torch.transpose(ensrc, 0, 1)
        desrc = x[desrc_idx].reshape(1,-1,self.node_dim)
        memory = self.transformerencoder(ensrc)
        memory = memory * sign
        tgt = self.transformerdecoder(desrc,memory)
        memory_temp = torch.transpose(memory, 0, 1)
        memory_temp = torch.transpose(memory_temp, 1, 2)
        tgt_temp = torch.transpose(tgt,0,1)
        att = torch.bmm(tgt_temp, memory_temp)
        att = self.softmax(att.squeeze(1))
        m = Categorical(att)
        choice = m.sample()
        cnt = self.perc * actual_amount[torch.arange(batch_size), choice]
        '''
        if(actual_amount[choice,0][0]>0):
            cnt = self.perc*actual_amount[choice,0]
        else:
            cnt = self.perc*actual_amount[choice,1]
        '''
        return choice, cnt, att

    def value(self, x, *args):
        x = torch.transpose(x.reshape(-1, self.per_graph_size, self.node_dim), 0, 1)
        src_dest = args[-1]
        batch_size = src_dest.shape[0]
        src = src_dest[:,0].long()
        dest = src_dest[:,1].long()
        memory = self.critic_encoder(x)
        src_mem = memory[src,torch.arange(batch_size)]
        dest_mem = memory[dest,torch.arange(batch_size)]
        # src_mem = x[src,torch.arange(batch_size)]
        # dest_mem = x[dest,torch.arange(batch_size)]
        last_feature = torch.cat((src_mem,dest_mem),-1)
        output = self.critic_mlp(last_feature)
        return output.squeeze(-1)