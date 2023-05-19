# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    def __init__(
        self, in_feats: int, out_feats: int, norm: bool=True, jump: bool=True, bias: bool=True, activation=None,
    ) -> None:
        """The GraphConvLayer that can forward based on the given graph and graph node features.

        Args:
            in_feats (int): The dimension of input feature.
            out_feats (int): The dimension of the output tensor.
            norm (bool): Add feature normalization operation or not. Defaults to True.
            jump (bool): Add skip connections of the input feature to the aggregation or not. Defaults to True.
            bias (bool): Add a learnable bias layer or not. Defaults to True.
            activation (torch.nn.functional): The output activation function to use. Defaults to None.
        """
        super(GraphConvLayer, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._jump = jump
        self._activation = activation

        if jump:
            self.weight = nn.Parameter(torch.Tensor(2 * in_feats, out_feats))
        else:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, feat: torch.Tensor, graph: dgl.DGLGraph, mask=None) -> torch.Tensor:
        if self._jump:
            _feat = feat

        if self._norm:
            if mask is None:
                norm = torch.pow(graph.in_degrees().float(), -0.5)
                norm.masked_fill_(graph.in_degrees() == 0, 1.0)
                shp = norm.shape + (1,) * (feat.dim() - 1)
                norm = torch.reshape(norm, shp).to(feat.device)
                feat = feat * norm.unsqueeze(1)
            else:
                graph.ndata["h"] = mask.float()
                graph.update_all(fn.copy_u(u="h", out="m"), fn.sum(msg="m", out="h"))
                masked_deg = graph.ndata.pop("h")
                norm = torch.pow(masked_deg, -0.5)
                norm.masked_fill_(masked_deg == 0, 1.0)
                feat = feat * norm.unsqueeze(-1)

        if mask is not None:
            feat = mask.float().unsqueeze(-1) * feat

        graph.ndata["h"] = feat
        graph.update_all(fn.copy_u(u="h", out="m"), fn.sum(msg="m", out="h"))
        rst = graph.ndata.pop("h")

        if self._norm:
            rst = rst * norm.unsqueeze(-1)

        if self._jump:
            rst = torch.cat([rst, _feat], dim=-1)

        rst = torch.matmul(rst, self.weight)

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst


class GraphConvNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation=F.relu,
        out_activation=None,
    ) -> None:
        """The GraphConvNet constructed with multiple GraphConvLayers.

        Args:
            input_dim (int): The dimension of the input feature.
            output_dim (int): The dimension of the output tensor.
            hidden_dim (int): The dimension of the hidden layers.
            num_layers (int): How many layers in this GraphConvNet in total, including the input and output layer. >= 2.
            activation (torch.nn.functional): The activation function used in input layer and hidden layers. Defaults to
                torch.nn.functional.relu.
            out_activation (torch.nn.functional): The output activation function to use. Defaults to None.
        """
        super(GraphConvNet, self).__init__()

        self.layers = nn.ModuleList(
            [GraphConvLayer(input_dim, hidden_dim, activation=activation)]
            + [GraphConvLayer(hidden_dim, hidden_dim, activation=activation) for _ in range(num_layers - 2)]
            + [GraphConvLayer(hidden_dim, output_dim, activation=out_activation)]
        )

    def forward(self, h: torch.Tensor, graph: dgl.DGLGraph, mask=None) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, graph, mask=mask)
        return h
