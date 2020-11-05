# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.simulator.utils.common import get_available_envs, get_scenarios, get_topologies


# maro env list
def list_available():
    """
    Show available env configurations in package

    Args:
        None

    Returns:
        None
    """
    envs = get_available_envs()

    for env in envs:
        print(f'scenario: {env["scenario"]}, topology: {env["topology"]}')


# maro env list
def list_scenarios(**kwargs):
    """
    Show all avaiable scenarios
    """

    for scenario in get_scenarios():
        print(scenario)


# maro env topologies --name scenario
def list_topologies(scenario: str, **kwargs):
    """
    Show topologies for specified scenario
    """

    for topology in get_topologies(scenario):
        print(topology)
