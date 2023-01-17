from typing import List, Type

import torch


def mlp(
    sizes: List[int],
    activation: Type[torch.nn.Module],
    output_activation: Type[torch.nn.Module] = torch.nn.Identity,
) -> torch.nn.Sequential:
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[j], sizes[j + 1]), act()]
    return torch.nn.Sequential(*layers)
