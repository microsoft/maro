from nn_builder.pytorch.NN import NN


def create_NN(input_dim: int, output_dim: int, hyperparameters: dict, seed: int, device: str):
    """Creates a neural network for the agents to use"""
    return NN(
        input_dim=input_dim,
        layers_info=hyperparameters["linear_hidden_units"] + [output_dim],
        output_activation=hyperparameters["output_activation"],
        hidden_activations=hyperparameters["hidden_activations"],
        dropout=hyperparameters["dropout"],
        initialiser=hyperparameters["initialiser"],
        batch_norm=hyperparameters["batch_norm"],
        random_seed=seed
    ).to(device)
