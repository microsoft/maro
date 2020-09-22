import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransGAT, PositionalEncoder, GeLU, Norm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.dropout import Dropout
# import numpy as np
# import datetime
class Header(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, net_type='res'):
        super().__init__()
        self.net_type = net_type
        if net_type == 'res':
            self.fc_0 = nn.Linear(input_size, hidden_size)
            self.act_0 = GeLU()
            # self.do_0 = Dropout(dropout)
            self.fc_1 = nn.Linear(hidden_size, input_size)
            self.act_1 = GeLU()
            self.fc_2 = nn.Linear(input_size, output_size)
        elif net_type == '2layer':
            self.fc_0 = nn.Linear(input_size, hidden_size)
            self.act_0 = GeLU()
            # self.do_0 = Dropout(dropout)
            self.fc_1 = nn.Linear(hidden_size, hidden_size//2)
            self.act_1 = GeLU()
            self.fc_2 = nn.Linear(hidden_size//2, output_size)
        elif net_type == '1layer':
            self.fc_0 = nn.Linear(input_size, hidden_size)
            self.act_0 = GeLU()
            self.fc_1 = nn.Linear(hidden_size, output_size)

    
    def forward(self, x):
        if self.net_type == 'res':
            x1 = self.act_0(self.fc_0(x))
            x1 = self.act_1(self.fc_1(x1) + x)
            return self.fc_2(x1)
        elif self.net_type == '2layer':
            x = self.act_0(self.fc_0(x))
            x = self.act_1(self.fc_1(x))
            x = self.fc_1(x)
            return x
        else:
            x = self.fc_1(self.act_0(self.fc_0(x)))
            return x


class SharedAC(nn.Module):
    def __init__(self, input_dim_p, edge_dim_p, input_dim_v, edge_dim_v, tick_buffer, action_dim, a=True, c=True, scale=4, ac_head='res'):
        super().__init__()
        assert(a or c)
        self.a, self.c = a, c
        self.input_dim_v = input_dim_v
        self.input_dim_p = input_dim_p
        self.tick_buffer = tick_buffer

        self.pre_dim_v, self.pre_dim_p = 8*scale, 16*scale
        self.p_pre_layer = nn.Sequential(nn.Linear(input_dim_p, self.pre_dim_p), GeLU(), PositionalEncoder(d_model=self.pre_dim_p, max_seq_len=tick_buffer))
        self.v_pre_layer = nn.Sequential(nn.Linear(input_dim_v, self.pre_dim_v), GeLU(), PositionalEncoder(d_model=self.pre_dim_v, max_seq_len=tick_buffer))
        p_encoder_layer = TransformerEncoderLayer(d_model=self.pre_dim_p, nhead=4, activation='gelu', dim_feedforward=self.pre_dim_p*4)
        v_encoder_layer = TransformerEncoderLayer(d_model=self.pre_dim_v, nhead=2, activation='gelu', dim_feedforward=self.pre_dim_v*4)
        # self.trans_layer_p = TransformerEncoder(p_encoder_layer, num_layers=3, norm=Norm(self.pre_dim_p))
        # self.trans_layer_v = TransformerEncoder(v_encoder_layer, num_layers=3, norm=Norm(self.pre_dim_v))
        self.trans_layer_p = TransformerEncoder(p_encoder_layer, num_layers=3)
        self.trans_layer_v = TransformerEncoder(v_encoder_layer, num_layers=3)
        
        self.gnn_output_size = 32*scale
        self.trans_gat = TransGAT(
            p_dim=self.pre_dim_p,
            v_dim=self.pre_dim_v,
            output_size=self.gnn_output_size//2,
            edge_dim={'p': edge_dim_p, 'v': edge_dim_v},
            layer_num=2
        )

        # self.reduce_dim = nn.Linear(self.a_input, 2)

        if a:
            self.policy_hidden_size = 16*scale
            self.a_input = 3*self.gnn_output_size//2
            self.actor = nn.Sequential(Header(self.a_input, self.policy_hidden_size, action_dim, ac_head), nn.Softmax(dim=-1))
        if c:
            self.value_hidden_size = 16*scale
            self.c_input = self.gnn_output_size
            self.critic = Header(self.c_input, self.value_hidden_size, 1, ac_head)
        

    def forward(self, state, a=False, p_idx=None, v_idx=None, c=False):
        assert((a and p_idx is not None and v_idx is not None) or c)
        feature_p, feature_v = state['p'], state['v']
        
        tb, bsize, p_cnt, _ = feature_p.shape
        v_cnt = feature_v.shape[2]
        assert(tb == self.tick_buffer)

        # before: feature_p.shape: (tick_buffer, batch_size, p_cnt, p_dim)
        # after: feature_p.shape: (tick_buffer, batch_size*p_cnt, p_dim)
        feature_p = self.p_pre_layer(feature_p.reshape(feature_p.shape[0], -1, feature_p.shape[-1]))
        # state['mask']: (batch_size, tick_buffer)
        # mask_p: (batch_size, p_cnt, tick_buffer)
        mask_p = state['mask'].repeat(1, p_cnt).reshape(-1, self.tick_buffer)
        feature_p = self.trans_layer_p(feature_p, src_key_padding_mask=mask_p)

        feature_v = self.v_pre_layer(feature_v.reshape(feature_v.shape[0], -1, feature_v.shape[-1]))
        mask_v = state['mask'].repeat(1, v_cnt).reshape(-1, self.tick_buffer)
        feature_v = self.trans_layer_v(feature_v, src_key_padding_mask=mask_v)

        feature_p = feature_p[0].reshape(bsize, p_cnt, self.pre_dim_p)
        feature_v = feature_v[0].reshape(bsize, v_cnt, self.pre_dim_v)

        emb_p, emb_v = self.trans_gat(feature_p, state['pe'], feature_v, state['ve'], state['ppe'])
        
        # date_str = f"{datetime.datetime.now().strftime('%Y%m%d')}"
        # time_str = f"{datetime.datetime.now().strftime('%H%M%S.%f')}"
        # subfolder_name = '%s_%s'%('./visualization/graph_embedding_p.npy', time_str)
        # np.save('%s_%s.npy'%('./visualization/emb/p_emb/emb', time_str), emb_p.cpu().detach().numpy())
        # np.save('%s_%s.npy'%('./visualization/emb/v_emb/emb', time_str), emb_v.cpu().detach().numpy())
        
        a_rtn, c_rtn = None, None
        if a and self.a:
            ap = emb_p.reshape(bsize, p_cnt, self.gnn_output_size)
            ap = ap[:, p_idx, :]
            av = emb_v.reshape(bsize, v_cnt, self.gnn_output_size//2)
            av = av[:, v_idx, :]
            emb_a = torch.cat((ap, av), axis=1)
            a_rtn = self.actor(emb_a)
        if c and self.c:
            c_rtn = self.critic(emb_p).reshape(bsize, p_cnt)
        return a_rtn, c_rtn
        
        