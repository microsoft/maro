import torch
from torch import nn
from torch.distributions import Categorical

from examples.citi_bike.ppo.models.transformer import (TransformerDecoder, TransformerEncoder, TransformerEncoderLayer,
                                                       CustomTransformerDecoderLayer)
from examples.citi_bike.ppo.models.homo_gnn import MultiChannelLinear


class AttTransPolicy(nn.Module):
    def __init__(self, node_dim, neighbor_cnt, per_graph_size):
        super().__init__()
        self.node_dim = node_dim
        self.softmax = nn.Softmax(-1)
        self.per_graph_size = per_graph_size
        self.neighbor_cnt = neighbor_cnt
        self.perc = 0.5
        self.amt_resolution = 11
        self.amt_max = 6
        self.amt_step = self.amt_max / (self.amt_resolution - 1)
        self.encoderLayer = TransformerEncoderLayer(d_model=self.node_dim, nhead=2, dropout=0)
        self.transformerencoder = TransformerEncoder(self.encoderLayer, 1)
        self.decoderLayer = CustomTransformerDecoderLayer(d_model=self.node_dim, nhead=2, dropout=0)
        self.transformerdecoder = TransformerDecoder(self.decoderLayer, 1)

        self.amt_hidden = 32
        self.amt_mask_arange = self.amt_step * torch.arange(self.amt_resolution, dtype=torch.float, requires_grad=False)
        self.amt_header = nn.Sequential(nn.Linear(self.node_dim*2, self.amt_hidden), nn.ReLU(), nn.Linear(self.amt_hidden, self.amt_resolution))
        self.amt_softmax = nn.Softmax(-1)

        '''
        critic_encoder_layer = TransformerEncoderLayer(d_model=self.node_dim, nhead=2)
        self.critic_transformer = TransformerEncoder(critic_encoder_layer, 1)
        self.critic_header = nn.Sequential(nn.Linear(self.node_dim, 16), nn.ReLU(), nn.Linear(16, 1))
        '''

        self.critic_hidden_dim = 16
        self.critic_headers = nn.Sequential(MultiChannelLinear(self.per_graph_size, self.node_dim, self.critic_hidden_dim), nn.ReLU(), MultiChannelLinear(self.per_graph_size, self.critic_hidden_dim, 1))

    def forward(self, x, edge_index, actual_amount, real_choice=None,noise_scale=0.0):
        # calculation attention
        row, col = edge_index
        # print("col",col)
        # get the source group
        actual_amount = torch.sum(actual_amount.reshape(-1, self.neighbor_cnt+1, 2), axis=-1)
        batch_size = actual_amount.shape[0]
        sign = actual_amount.new_ones((batch_size, 1), dtype=torch.int, requires_grad=False)
        sign[actual_amount[:, 0] < 0, 0] = -1

        desrc_idx = col.reshape(-1, self.neighbor_cnt + 1)[:, -1].reshape(-1)
        # ensrc_idx = col.reshape(-1, self.neighbor_cnt+1)[:, :-1].reshape(-1)
        ensrc = x[col].reshape(-1, self.neighbor_cnt + 1, self.node_dim)
        ensrc = torch.transpose(ensrc, 0, 1)
        desrc = x[desrc_idx].reshape(1, -1, self.node_dim)
        memory = self.transformerencoder(ensrc)
        memory = memory * sign
        tgt = self.transformerdecoder(desrc, memory)
        memory1 = torch.transpose(memory, 0, 1)
        memory2 = torch.transpose(memory1, 1, 2)

        tgt_temp = torch.transpose(tgt, 0, 1)
        att = torch.bmm(tgt_temp, memory2)
        att = self.softmax(att.squeeze(1))
        m = Categorical(att)
        if(real_choice is None):
            choice = m.sample()
        else:
            choice = real_choice.reshape(-1)
        batch_idx = torch.arange(batch_size, requires_grad=False).to(choice.device)

        # chosen_dest.shape: B*D
        chosen_dest = memory1[batch_idx, choice]
        amt_input = torch.cat((chosen_dest, tgt.reshape(batch_size, self.node_dim)), axis=1)
        amt_ratio = self.amt_header(amt_input)

        # chosen_actual_amount: B
        chosen_actual_amount = actual_amount[batch_idx, choice].reshape(-1, 1)
        batch_amt_arange = self.amt_mask_arange.to(chosen_actual_amount.device).repeat(batch_size, 1)
        abs_chosen_actual_amount = torch.abs(chosen_actual_amount)

        amt_ratio[batch_amt_arange > abs_chosen_actual_amount] = float('-inf')
        # disable learning of this header
        '''
        amt_ratio[batch_amt_arange < (abs_chosen_actual_amount/2)] = float('-inf')
        amt_ratio[batch_idx, (abs_chosen_actual_amount/self.amt_step).long()] = 0
        '''
        # print(abs_chosen_actual_amount, amt_ratio)
        # print(abs_chosen_actual_amount, amt_ratio)
        amt_ratio = self.amt_softmax(amt_ratio)

        amt_m = Categorical(amt_ratio)
        amt_choice = amt_m.sample()

        # cnt = self.perc * actual_amount[batch_idx, choice]
        return choice, att, amt_choice, amt_ratio

    def value(self, x, *args):
        '''
        x = torch.transpose(x.reshape(-1, self.per_graph_size, self.node_dim), 0, 1)
        emb = self.critic_transformer(x)
        values = self.critic_header(emb.reshape(-1, self.node_dim))
        values = torch.transpose(values.reshape(self.per_graph_size, -1), 0, 1)
        return values

        '''
        x = torch.transpose(x.reshape(-1, self.per_graph_size, self.node_dim), 0, 1)
        output = self.critic_headers(x).reshape(self.per_graph_size, -1)
        return torch.transpose(output, 0, 1)
