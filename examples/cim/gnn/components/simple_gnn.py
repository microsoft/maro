import math

import torch
import torch.nn as nn
from torch import Tensor

from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm

from maro.rl import AbsLearningModel, OptimizerOptions


class PositionalEncoder(nn.Module):
    """
    The positional encoding used in transformer to get the sequential information.

    The code is based on the PyTorch version in web
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html?highlight=positionalencoding
    """

    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model
        self.times = 4 * math.sqrt(self.d_model)

        # Create constant "pe" matrix with values dependent on pos and i.
        self.pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                self.pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                self.pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        self.pe = self.pe.unsqueeze(1) / self.d_model

    def forward(self, x):
        # Make embeddings relatively larger.
        addon = self.pe[: x.shape[0], :, : x.shape[2]].to(x.device)
        return x + addon


class GeLU(nn.Module):
    """Simple gelu wrapper as a independent module."""
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return F.gelu(input)


class SimpleGATLayer(nn.Module):
    """The enhanced graph attention layer for heterogenenous neighborhood.

    It first utilizes pre-layers for both the source and destination node to map their features into the same hidden
    size. If the edge also has features, they are concatenated with those of the corresponding source node before being
    fed to the pre-layers. Then the graph attention(https://arxiv.org/abs/1710.10903) is done to aggregate information
    from the source nodes to the destination nodes. The residual connection and layer normalization are also used to
    enhance the performance, which is similar to the Transformer(https://arxiv.org/abs/1706.03762).

    Args:
        src_dim (int): The feature dimension of the source nodes.
        dest_dim (int): The feature dimension of the destination nodes.
        edge_dim (int): The feature dimension of the edges. If the edges have no feature, it should be set 0.
        hidden_size (int): The hidden size both the destination and source is mapped into.
        nhead (int): The number of head in the multi-head attention.
        position_encoding (bool): the neighbor source nodes is aggregated in order(True) or orderless(False).
    """

    def __init__(self, src_dim, dest_dim, edge_dim, hidden_size, nhead=4, position_encoding=True):
        super().__init__()
        self.src_dim = src_dim
        self.dest_dim = dest_dim
        self.edge_dim = edge_dim
        self.hidden_size = hidden_size
        self.nhead = nhead
        src_layers = []
        src_layers.append(nn.Linear(src_dim + edge_dim, hidden_size))
        src_layers.append(GeLU())
        self.src_pre_layer = nn.Sequential(*src_layers)

        dest_layers = []
        dest_layers.append(nn.Linear(dest_dim, hidden_size))
        dest_layers.append(GeLU())
        self.dest_pre_layer = nn.Sequential(*dest_layers)

        self.att = MultiheadAttention(embed_dim=hidden_size, num_heads=nhead)
        self.att_dropout = Dropout(0.1)
        self.att_norm = LayerNorm(hidden_size)

        self.zero_padding_template = torch.zeros((1, src_dim), dtype=torch.float)

    def forward(self, src: Tensor, dest: Tensor, adj: Tensor, mask: Tensor, edges: Tensor = None):
        """Information aggregation from the source nodes to the destination nodes.

        Args:
            src (Tensor): The source nodes in a batch of graph.
            dest (Tensor): The destination nodes in a batch of graph.
            adj (Tensor): The adjencency list stored in a 2D matrix in the batch-second format. The first dimension is
                the maximum amount of the neighbors the destinations have. As the neighbor quantities vary from one
                destination to another, the short sequences are padded with 0.
            mask (Tensor): The mask identifies if a position in the adj is padded. Note that it is stored in the
                batch-first format.

        Returns:
            destination_emb: The embedding of the destinations after the GAT layer.

        Shape:
            src: (batch, src_cnt, src_dim)
            dest: (batch, dest_cnt, dest_dim)
            adj: (src_neighbor_cnt, batch*dest_cnt)
            mask: (batch*dest_cnt)*src_neighbor_cnt
            edges: (batch*dest_cnt, src_neighbor_cnt, edge_dim)
            destination_emb: (batch, dest_cnt, hidden_size)

        """
        assert(self.src_dim == src.shape[-1])
        assert(self.dest_dim == dest.shape[-1])
        batch, s_cnt, src_dim = src.shape
        batch, d_cnt, dest_dim = dest.shape
        src_neighbor_cnt = adj.shape[0]

        src_embedding = src.reshape(-1, src_dim)
        src_embedding = torch.cat((self.zero_padding_template.to(src_embedding.device), src_embedding))

        flat_adj = adj.reshape(-1)
        src_embedding = src_embedding[flat_adj].reshape(src_neighbor_cnt, -1, src_dim)
        if edges is not None:
            src_embedding = torch.cat((src_embedding, edges), axis=2)

        src_input = self.src_pre_layer(
            src_embedding.reshape(-1, src_dim + self.edge_dim)). \
            reshape(*src_embedding.shape[:2], self.hidden_size)
        dest_input = self.dest_pre_layer(dest.reshape(-1, dest_dim)).reshape(1, batch * d_cnt, self.hidden_size)
        dest_emb, _ = self.att(dest_input, src_input, src_input, key_padding_mask=mask)

        dest_emb = dest_emb + self.att_dropout(dest_emb)
        dest_emb = self.att_norm(dest_emb)
        return dest_emb.reshape(batch, d_cnt, self.hidden_size)


class SimpleTransformer(nn.Module):
    """Graph attention network with multiple graphs in the CIM scenario.

    This module aggregates information in the port-to-port graph, port-to-vessel graph and vessel-to-port graph. The
    aggregation in the two graph are done separatedly and then the port features are concatenated as the final result.

    Args:
        p_dim (int): The feature dimension of the ports.
        v_dim (int): The feature dimension of the vessels.
        edge_dim (dict): The key is the edge name and the value is the corresponding feature dimension.
        output_size (int): The hidden size in graph attention.
        layer_num (int): The number of graph attention layers in each graph.
    """

    def __init__(self, p_dim, v_dim, edge_dim: dict, output_size, layer_num=2):
        super().__init__()
        self.hidden_size = output_size
        self.layer_num = layer_num

        pl, vl, ppl = [], [], []
        for i in range(layer_num):
            if i == 0:
                pl.append(SimpleGATLayer(v_dim, p_dim, edge_dim["v"], self.hidden_size, nhead=4))
                vl.append(SimpleGATLayer(p_dim, v_dim, edge_dim["v"], self.hidden_size, nhead=4))
                # p2p links.
                ppl.append(
                    SimpleGATLayer(
                        p_dim, p_dim, edge_dim["p"], self.hidden_size, nhead=4, position_encoding=False)
                )
            else:
                pl.append(SimpleGATLayer(self.hidden_size, self.hidden_size, 0, self.hidden_size, nhead=4))
                if i != layer_num - 1:
                    # p2v conv is not necessary at the last layer, for we only use port features.
                    vl.append(SimpleGATLayer(self.hidden_size, self.hidden_size, 0, self.hidden_size, nhead=4))
                ppl.append(SimpleGATLayer(
                    self.hidden_size, self.hidden_size, 0, self.hidden_size, nhead=4, position_encoding=False))
        self.p_layers = nn.ModuleList(pl)
        self.v_layers = nn.ModuleList(vl)
        self.pp_layers = nn.ModuleList(ppl)

    def forward(self, p, pe, v, ve, ppe):
        """Do the multi-channel graph attention.

        Args:
            p (Tensor): The port feature.
            pe (Tensor): The vessel-port edge feature.
            v (Tensor): The vessel feature.
            ve (Tensor): The port-vessel edge feature.
            ppe (Tensor): The port-port edge feature.
        """
        # p.shape: (batch*p_cnt, p_dim)
        pp = p
        pre_p, pre_v, pre_pp = p, v, pp
        for i in range(self.layer_num):
            # Only feed edge info in the first layer.
            p = self.p_layers[i](pre_v, pre_p, adj=pe["adj"], edges=pe["edge"] if i == 0 else None, mask=pe["mask"])
            if i != self.layer_num - 1:
                v = self.v_layers[i](
                    pre_p, pre_v, adj=ve["adj"], edges=ve["edge"] if i == 0 else None, mask=ve["mask"])
            pp = self.pp_layers[i](
                pre_pp, pre_pp, adj=ppe["adj"], edges=ppe["edge"] if i == 0 else None, mask=ppe["mask"])
            pre_p, pre_v, pre_pp = p, v, pp
        p = torch.cat((p, pp), axis=2)
        return p, v


class GNNBasedACModel(AbsLearningModel):
    """The actor-critic module shared with multiple agents.

    This module maps the input graph of the observation to the policy and value space. It first extracts the temporal
    information separately for each node with a small transformer block and then extracts the spatial information with
    a multi-graph/channel graph attention. Finally, the extracted feature embedding is fed to a actor header as well
    as a critic layer, which are the two MLPs with residual connections.
    """

    def __init__(self, component, p_pre_dim, v_pre_dim, sequence_buffer_size, gnn_output_size, optimizer_options=None):
        super().__init__(component, optimizer_options=optimizer_options)
        self.p_pre_dim = p_pre_dim
        self.v_pre_dim = v_pre_dim
        self.sequence_buffer_size = sequence_buffer_size
        self.gnn_output_size = gnn_output_size

    def forward(self, state, actor_enabled=False, critic_enabled=False, is_training=True):
        self.train(mode=is_training)
        if is_training:
            return self._forward(state, actor_enabled=actor_enabled, critic_enabled=critic_enabled)

        with torch.no_grad():
            return self._forward(state, actor_enabled=actor_enabled, critic_enabled=critic_enabled)

    def _forward(self, state, actor_enabled=False, critic_enabled=False):
        p_idx, v_idx = state.get("p_idx", None), state.get("v_idx", None)
        assert((actor_enabled and p_idx is not None and v_idx is not None) or critic_enabled)
        feature_p, feature_v = state["p"].float(), state["v"].float()
        tb, bsize, p_cnt, _ = feature_p.shape
        v_cnt = feature_v.shape[2]
        assert(tb == self.sequence_buffer_size)

        # Before: feature_p.shape: (tick_buffer, batch_size, p_cnt, p_dim)
        # After: feature_p.shape: (tick_buffer, batch_size*p_cnt, p_dim)
        feature_p = self._component["p_pre_layers"](feature_p.reshape(feature_p.shape[0], -1, feature_p.shape[-1]))
        # state["mask"]: (batch_size, tick_buffer)
        # mask_p: (batch_size, p_cnt, tick_buffer)
        mask_p = state["mask"].repeat(1, p_cnt).reshape(-1, self.sequence_buffer_size)
        feature_p = self._component["p_trans_layers"](feature_p, src_key_padding_mask=mask_p)

        feature_v = self._component["v_pre_layers"](feature_v.reshape(feature_v.shape[0], -1, feature_v.shape[-1]))
        mask_v = state["mask"].repeat(1, v_cnt).reshape(-1, self.sequence_buffer_size)
        feature_v = self._component["v_trans_layers"](feature_v, src_key_padding_mask=mask_v)

        feature_p = feature_p[0].reshape(bsize, p_cnt, self.p_pre_dim)
        feature_v = feature_v[0].reshape(bsize, v_cnt, self.v_pre_dim)

        emb_p, emb_v = self._component["trans_gat"](feature_p, state["pe"], feature_v, state["ve"], state["ppe"])

        a_rtn, c_rtn = None, None
        if actor_enabled and "actor_head" in self._component:
            ap = emb_p.reshape(bsize, p_cnt, self.gnn_output_size)
            ap = ap[:, p_idx, :]
            av = emb_v.reshape(bsize, v_cnt, self.gnn_output_size // 2)
            av = av[:, v_idx, :]
            emb_a = torch.cat((ap, av), axis=1)
            a_rtn = self._component["actor_head"](emb_a)
        if critic_enabled and "critic_head" in self._component:
            c_rtn = self._component["critic_head"](emb_p).reshape(bsize, p_cnt)
        
        return a_rtn, c_rtn
