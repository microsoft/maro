# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from typing import Dict, List


def _choose_by_weight(graph: Dict[int, List[int]], node2weight: Dict[int, float]) -> List[int]:
    """Choose node in the order of descending weight if not blocked..

    Args:
        graph (Dict[int, List[int]]): The adjacent matrix of the target graph. The key is the node id of each node. The
            value is a list of each node's neighbor nodes.
        node2weight (Dict[int, float]): The node to weight dictionary with node id as key and node weight as value.

    Returns:
        List[int]: A list of chosen node id.
    """
    node_weight_list = [(node, weight) for node, weight in node2weight.items()]
    # Shuffle the candidates to get random result in the case there are nodes sharing the same weight.
    random.shuffle(node_weight_list)
    # Sort node candidates with descending weight.
    sorted_nodes = sorted(node_weight_list, key=lambda x: x[1], reverse=True)

    chosen_node_id_set: set = set()
    blocked_node_id_set: set = set()
    # Choose node in the order of descending weight if it is not blocked yet by the chosen nodes.
    for node, _ in sorted_nodes:
        if node in blocked_node_id_set:
            continue
        chosen_node_id_set.add(node)
        for neighbor_node in graph[node]:
            blocked_node_id_set.add(neighbor_node)

    chosen_node_ids = [node for node in chosen_node_id_set]
    return chosen_node_ids

def uniform_mis_solver(graph: Dict[int, List[int]]) -> List[int]:
    node2weight: Dict[int, float] = {node: 1 for node in graph.keys()}
    chosen_node_list = _choose_by_weight(graph, node2weight)
    return chosen_node_list

def greedy_mis_solver(graph: Dict[int, List[int]]) -> List[int]:
    node2weight: Dict[int, float] = {node: 1 / (1 + len(neighbor_list)) for node, neighbor_list in graph.items()}
    chosen_node_list = _choose_by_weight(graph, node2weight)
    return chosen_node_list
