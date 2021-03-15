# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch.nn import GELU, Sequential, TransformerEncoder, TransformerEncoderLayer

from maro.rl import FullyConnectedBlock
from maro.utils import DummyLogger

from examples.cim.ac_gnn.config import agent_config

from .gnn_based_actor_critic import GNNBasedActorCritic, GNNBasedActorCriticConfig
from .model import GNNBasedACModel, PositionalEncoder, SimpleTransformer


def get_gnn_agent(p_dim, v_dim, p2p_adj):
    scale = agent_config["model"]["scale"]
    pedge_dim = agent_config["model"]["port_edge_feature_dim"]
    p_pre_dim = agent_config["model"]["port_pre_dim"] * scale
    vedge_dim = agent_config["model"]["vessel_edge_feature_dim"]
    v_pre_dim = agent_config["model"]["vessel_pre_dim"] * scale
    sequence_buffer_size = agent_config["model"]["sequence_buffer_size"]
    gnn_output_size = agent_config["model"]["graph_output_dim"] * scale
    actor_input_dim = 3 * gnn_output_size // 2
    model = GNNBasedACModel(
        {
        "p_pre_layers": 
            Sequential(
                FullyConnectedBlock(p_dim, p_pre_dim, [], activation=GELU),
                PositionalEncoder(d_model=p_pre_dim, max_seq_len=sequence_buffer_size)
            ), 
        "v_pre_layers": 
            Sequential(
                FullyConnectedBlock(v_dim, v_pre_dim, [], activation=GELU),
                PositionalEncoder(d_model=v_pre_dim, max_seq_len=sequence_buffer_size)
            ),
        "p_trans_layers": 
            TransformerEncoder(
                TransformerEncoderLayer(d_model=p_pre_dim, nhead=4, activation="gelu", dim_feedforward=p_pre_dim * 4),
                num_layers=3
            ),
        "v_trans_layers":
            TransformerEncoder(
                TransformerEncoderLayer(d_model=v_pre_dim, nhead=2, activation="gelu", dim_feedforward=v_pre_dim * 4),
                num_layers=3
            ),
        "trans_gat":
            SimpleTransformer(
                p_dim=p_pre_dim,
                v_dim=v_pre_dim,
                output_size=gnn_output_size // 2,
                edge_dim={"p": pedge_dim, "v": vedge_dim},
                layer_num=2
            ),
        "actor_head":
            FullyConnectedBlock(
                actor_input_dim, 
                agent_config["model"]["action_dim"],
                [d * scale for d in agent_config["model"]["policy_hidden_dims"]] + [actor_input_dim],
                activation=GELU,
                head=True,
                softmax=True
            ),
        "critic_head":
            FullyConnectedBlock(
                gnn_output_size,
                1,
                [d * scale for d in agent_config["model"]["value_hidden_dims"]] + [gnn_output_size],
                head=True,
                activation=GELU
            )
        },
        sequence_buffer_size=sequence_buffer_size,
        optim_option=agent_config["optimization"]
    )

    return GNNBasedActorCritic(model, GNNBasedActorCriticConfig(p2p_adj=p2p_adj, **agent_config["hyper_params"]))
