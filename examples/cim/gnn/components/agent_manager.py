import os
from collections import defaultdict
from copy import copy

import numpy as np
import torch
from torch.nn import GELU, TransformerEncoder, TransformerEncoderLayer
from torch.optim import Adam

from maro.rl import AbsAgentManager, ActionInfo, FullyConnectedBlock, NNStack, OptimizerOptions
from maro.utils import DummyLogger, Logger

from examples.cim.gnn.components.gnn_based_actor_critic import GNNBasedActorCritic, GNNBasedActorCriticConfig
from examples.cim.gnn.components.numpy_store import NumpyStore
from examples.cim.gnn.components.simple_gnn import GNNBasedACModel, PositionalEncoder, SimpleTransformer
from examples.cim.gnn.components.state_shaper import GNNStateShaper
from examples.cim.gnn.general import training_logger


def get_experience_pool(config):
    num_static_nodes, num_dynamic_nodes = config.num_static_nodes, config.num_dynamic_nodes
    p_dim, pedge_dim = config.model.port_feature_dim, config.model.port_edge_feature_dim
    v_dim, vedge_dim = config.model.vessel_feature_dim, config.model.vessel_edge_feature_dim
    sequence_buffer_size = config.model.sequence_buffer_size
    value_dict = {
        ("s", "v"): ((sequence_buffer_size, num_dynamic_nodes, v_dim), np.float32, False),
        ("s", "p"): ((sequence_buffer_size, num_static_nodes, p_dim), np.float32, False),
        ("s", "vo"): ((num_dynamic_nodes, num_static_nodes), np.int64, True),
        ("s", "po"): ((num_static_nodes, num_dynamic_nodes), np.int64, True),
        ("s", "vedge"): ((num_dynamic_nodes, num_static_nodes, vedge_dim), np.float32, True),
        ("s", "pedge"): ((num_static_nodes, num_dynamic_nodes, vedge_dim), np.float32, True),
        ("s", "ppedge"): ((num_static_nodes, num_static_nodes, pedge_dim), np.float32, True),
        ("s", "mask"): ((sequence_buffer_size,), np.bool, True),
        ("s_", "v"): ((sequence_buffer_size, num_dynamic_nodes, v_dim), np.float32, False),
        ("s_", "p"): ((sequence_buffer_size, num_static_nodes, p_dim), np.float32, False),
        ("s_", "vo"): ((num_dynamic_nodes, num_static_nodes), np.int64, True),
        ("s_", "po"): ((num_static_nodes, num_dynamic_nodes), np.int64, True),
        ("s_", "vedge"): ((num_dynamic_nodes, num_static_nodes, vedge_dim), np.float32, True),
        ("s_", "pedge"): ((num_static_nodes, num_dynamic_nodes, vedge_dim), np.float32, True),
        ("s_", "ppedge"): ((num_static_nodes, num_static_nodes, pedge_dim), np.float32, True),
        ("s_", "mask"): ((sequence_buffer_size,), np.bool, True),
        # To identify 1-dimensional variables.
        ("R",): ((num_static_nodes,), np.float32, True),
        ("a",): (tuple(), np.int64, True),
    }

    return {agent_id: NumpyStore(value_dict, size) for agent_id, size in config.exp_per_ep.items()}


def create_gnn_agent(config):
    training_logger.info(config.training.device)
    scale = config.model.scale
    p_dim, pedge_dim = config.model.port_feature_dim, config.model.port_edge_feature_dim
    v_dim, vedge_dim = config.model.vessel_feature_dim, config.model.vessel_edge_feature_dim
    p_pre_dim, v_pre_dim = config.model.port_pre_dim * scale, config.model.vessel_pre_dim * scale
    sequence_buffer_size = config.model.sequence_buffer_size
    gnn_output_size = config.model.graph_output_dim * scale
    actor_input_dim = 3 * gnn_output_size // 2
    model = GNNBasedACModel(
        p_pre_layers=NNStack(
            "static_node_pre_layers",
            FullyConnectedBlock(p_dim, p_pre_dim, [], activation=GELU),
            PositionalEncoder(d_model=p_pre_dim, max_seq_len=sequence_buffer_size)
        ), 
        v_pre_layers=NNStack(
            "dynamic_node_pre_layers",
            FullyConnectedBlock(v_dim, v_pre_dim, [], activation=GELU),
            PositionalEncoder(d_model=v_pre_dim, max_seq_len=sequence_buffer_size)
        ),
        p_trans_layers=NNStack(
            "static_node_transformer_encoder",
            TransformerEncoder(
                TransformerEncoderLayer(d_model=p_pre_dim, nhead=4, activation="gelu", dim_feedforward=p_pre_dim * 4),
                num_layers=3
            )
        ),
        v_trans_layers=NNStack(
            "dynamic_node_transformer_encoder",
            TransformerEncoder(
                TransformerEncoderLayer(d_model=v_pre_dim, nhead=2, activation="gelu", dim_feedforward=v_pre_dim * 4),
                num_layers=3
            )
        ),
        trans_gat=NNStack(
            "graph_attention_transformer",
            SimpleTransformer(
                p_dim=p_pre_dim,
                v_dim=v_pre_dim,
                output_size=gnn_output_size // 2,
                edge_dim={"p": pedge_dim, "v": vedge_dim},
                layer_num=2
            )
        ),
        actor_head=NNStack(
            "actor",
            FullyConnectedBlock(
                actor_input_dim, 
                config.model.action_dim,
                [d * scale for d in config.model.policy_hidden_dims] + [actor_input_dim],
                activation=GELU,
                is_head=True,
                softmax_enabled=True
            )
        ),
        critic_head=NNStack(
            "critic",
            FullyConnectedBlock(
                gnn_output_size,
                1,
                [d * scale for d in config.model.value_hidden_dims] + [gnn_output_size],
                is_head=True,
                activation=GELU
            )
        ),
        p_pre_dim=p_pre_dim,
        v_pre_dim=v_pre_dim,
        sequence_buffer_size=sequence_buffer_size,
        gnn_output_size=gnn_output_size,
        optimizer_options=OptimizerOptions(cls=Adam, params={"lr": config.model.learning_rate})
    )

    return GNNBasedActorCritic(
        "gnn-a2c", model, GNNBasedActorCriticConfig(**config.algorithm), 
        experience_pool=get_experience_pool(config),
        logger=training_logger
    )


class GNNAgentManager(AbsAgentManager):
    def choose_action(self, decision_event, snapshot_list):
        state = self._state_shaper(action_info=decision_event, snapshot_list=snapshot_list)
        action_info = self.agent.choose_action(state)
        self._experience_shaper.record(decision_event, action_info, state)
        return self._action_shaper(self._trajectory["action"][-1], decision_event)

    def on_env_feedback(self, metrics):
        pass

    def post_process(self, snapshot_list):
        experiences = self._experience_shaper(self._trajectory, snapshot_list)
        self._trajectory.clear()
        self._state_shaper.reset()
        self._action_shaper.reset()
        self._experience_shaper.reset()
        return experiences

    def store_experiences(self, experiences):
        self.agent.store_experiences(experiences)

    def train(self):
        self.agent.train()

    def dump_models_to_files(self, path):
        self.agent.dump_model_to_file(path)
