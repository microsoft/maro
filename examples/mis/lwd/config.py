# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


class Config(object):
    device: str = "cuda:0"

    # Configuration for graph batch size
    train_graph_batch_size = 32
    eval_graph_batch_size = 32

    # Configuration for num_samples
    train_num_samples = 2
    eval_num_samples = 10

    # Configuration for the MISEnv
    max_tick = 32  # Once the max_tick reached, the timeout processing will set all deferred nodes to excluded
    num_node_lower_bound: int = 15
    num_node_upper_bound: int = 20
    node_sample_probability: float = 0.15

    # Configuration for the reward definition
    diversity_reward_coef = 0.1  # reward = cardinality reward + coef * diversity Reward
    reward_normalization_base = 20

    # Configuration for the GraphBasedActorCritic
    input_dim = 2
    output_dim = 3
    hidden_dim = 128
    num_layers = 5

    # Configuration for PPO update
    init_lr = 1e-4
    clip_ratio = 0.2
    reward_discount = 1.0

    # Configuration for main loop
    batch_size = 16
    num_train_epochs = 4
