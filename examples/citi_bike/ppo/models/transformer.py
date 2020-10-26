import torch
import copy
import math

# lib for transformer
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
# import numpy as np
# from .transformer_source import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model
        self.times = 4 * math.sqrt(self.d_model)

        # create constant 'pe' matrix with values dependant on
        # pos and i
        self.pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                self.pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                self.pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        self.pe = self.pe.unsqueeze(1) / self.d_model

    def forward(self, x):
        # make embeddings relatively larger
        # x = x * self.sqrt_d_model
        # add constant to embedding
        # make the addon relatively smaller
        addon = self.pe[:x.shape[0], :, :x.shape[2]].to(x.get_device())
        return x + addon


class TransGATLayer(nn.Module):
    def __init__(self, src_dim, dest_dim, edge_dim, hidden_size, nhead=4, agreggate='first', position_encoding=True):
        super().__init__()
        self.src_dim = src_dim
        self.dest_dim = dest_dim
        self.edge_dim = edge_dim
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.concat_layers = []
        self.concat_layers.append(nn.Linear(src_dim + dest_dim + edge_dim, hidden_size))
        self.concat_layers.append(GeLU())
        self.enable_pe = position_encoding
        if position_encoding:
            self.pe = PositionalEncoder(d_model=hidden_size, max_seq_len=40)
        self.pre_layer = nn.Sequential(*self.concat_layers)

        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, activation='gelu',
                                                dim_feedforward=hidden_size * 4)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=2)
        self.aggregate_func = agreggate

        if self.aggregate_func == 'decoder':
            decoder_fc = [nn.Linear(dest_dim, hidden_size), GeLU()]
            self.decoder_fc = nn.Sequential(*decoder_fc)
            decoder_layer = CustomTransformerDecoderLayer(d_model=hidden_size, nhead=nhead, activation='gelu',
                                                          dim_feedforward=hidden_size * 4)
            self.decoder = TransformerDecoder(decoder_layer, num_layers=1)
        # self.encoder = TransformerEncoder(encoder_layer, num_layers=2)
        self.zero_padding_template = torch.zeros((1, src_dim), dtype=torch.float)

    def _aggregate(self, emb):
        # emb.shape: (seq_len, cnt, dim)
        if self.aggregate_func == 'first':
            return emb[0]
        elif self.aggregate_func == 'max_pooling':
            return torch.max(emb, axis=0)[0]
        elif self.aggregate_func == 'sum':
            return torch.sum(emb, axis=0)
        elif self.aggregate_func == 'mean':
            return torch.mean(emb, axis=0)

    def forward(self, src, dest, adj, mask, edges=None):
        '''
        src.shape: (batch, src_cnt, src_dim)
        dest.shape: (batch, dest_cnt, dest_dim)
        adj.shape (ordered): (src_cnt, batch*dest_cnt)
        mask.shape : (batch*dest_cnt)*src_cnt
        edges.shape: (batch*dest_cnt, src_cnt, edge_dim)
        '''
        assert(self.src_dim == src.shape[-1])
        assert(self.dest_dim == dest.shape[-1])
        batch, s_cnt, src_dim = src.shape
        batch, d_cnt, dest_dim = dest.shape

        src_embedding = src.reshape(-1, src.shape[-1])
        src_embedding = torch.cat((self.zero_padding_template.to(src_embedding.get_device()), src_embedding))

        # adj = adj[:,torch.randperm(adj.size()[1])]
        flat_adj = adj.reshape(-1)
        # src_cnt * (batch * dest_cnt) * src_dim
        dest_embedding = src_embedding[flat_adj].reshape(adj.shape[0], -1, src.shape[-1])

        if self.aggregate_func != 'decoder':
            if edges is not None:
                pre_input = torch.cat((dest_embedding, edges), axis=2)
        else:
            dest_rep = dest.reshape(-1, dest.shape[-1]).unsqueeze(0).repeat(adj.shape[0], 1, 1)
            # src_cnt * (batch * dest_cnt) * (src_dim + dest_dim + edge_dim*)
            if edges is None:
                pre_input = torch.cat((dest_embedding, dest_rep), axis=2)
            else:
                pre_input = torch.cat((dest_embedding, dest_rep, edges), axis=2)

        encoder_input = self.pre_layer(pre_input.reshape(-1, pre_input.shape[-1])).reshape(*pre_input.shape[:2],
                                                                                           self.hidden_size)
        if self.enable_pe:
            encoder_input = self.pe(encoder_input)

        # src_cnt * (batch*dest_cnt) * hidden_size
        dest_emb = self.encoder(encoder_input, src_key_padding_mask=mask)

        # only get the first dimension
        if self.aggregate_func == 'decoder':
            dest_decode = self.decoder_fc(dest.reshape(batch * d_cnt, dest_dim))
            aggregated_emb = self.decoder(dest_decode.reshape(1, batch * d_cnt, self.hidden_size), dest_emb,
                                          memory_key_padding_mask=mask)
        else:
            aggregated_emb = self._aggregate(dest_emb)
        return aggregated_emb.reshape(batch, d_cnt, self.hidden_size)


class TransGAT(nn.Module):
    def __init__(self, p_dim, v_dim, edge_dim: dict, output_size, layer_num=2):
        super().__init__()
        self.hidden_size = output_size
        self.layer_num = layer_num

        pl, vl, ppl = [], [], []
        for i in range(layer_num):
            if (i == 0):
                pl.append(TransGATLayer(v_dim, p_dim, edge_dim['v'], self.hidden_size, nhead=4, agreggate='decoder'))
                vl.append(TransGATLayer(p_dim, v_dim, edge_dim['v'], self.hidden_size, nhead=4, agreggate='decoder'))
                # p2p links
                ppl.append(TransGATLayer(p_dim, p_dim, edge_dim['p'], self.hidden_size, nhead=4, agreggate='decoder',
                                         position_encoding=False))
            else:
                pl.append(TransGATLayer(self.hidden_size, self.hidden_size, 0, self.hidden_size, nhead=4,
                                        agreggate='decoder'))
                if i != layer_num - 1:
                    # p2v conv is not necessary at the last layer, for we only use port features
                    vl.append(TransGATLayer(self.hidden_size, self.hidden_size, 0, self.hidden_size, nhead=4,
                                            agreggate='decoder'))
                ppl.append(TransGATLayer(self.hidden_size, self.hidden_size, 0, self.hidden_size, nhead=4,
                                         agreggate='decoder', position_encoding=False))
        self.p_layers = nn.ModuleList(pl)
        self.v_layers = nn.ModuleList(vl)
        self.pp_layers = nn.ModuleList(ppl)

    def forward(self, p, pe, v, ve, ppe):
        # p.shape: (batch*p_cnt, p_dim)
        pp = p
        pre_p, pre_v, pre_pp = p, v, pp
        for i in range(self.layer_num):
            # only feed edge info in the first layer
            p = self.p_layers[i](pre_v, pre_p, adj=pe['adj'], edges=pe['edge'] if i == 0 else None, mask=pe['mask'])
            if i != self.layer_num - 1:
                v = self.v_layers[i](pre_p, pre_v, adj=ve['adj'], edges=ve['edge'] if i == 0 else None, mask=ve['mask'])
            pp = self.pp_layers[i](pre_pp, pre_pp, adj=ppe['adj'], edges=ppe['edge'] if i == 0 else None,
                                   mask=ppe['mask'])
            pre_p, pre_v, pre_pp = p, v, pp
        p = torch.cat((p, pp), axis=2)
        return p, v


class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return F.gelu(input)


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class TransformerEncoder(Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output


class TransformerDecoder(Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](output, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, att = self.self_attn(src, src, src, attn_mask=src_mask,
                                   key_padding_mask=src_key_padding_mask)
        # np.save('./visualization/multi_attention.npy', att.cpu().detach().numpy())
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class CustomTransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(CustomTransformerDecoderLayer, self).__init__()
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        # self.linear2 = Linear(dim_feedforward, d_model)
        self.linear2 = Linear(dim_feedforward, d_model)

        # self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        # self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)
