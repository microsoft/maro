# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from functools import wraps

from maro.cli.utils.details import load_cluster_details
from maro.utils.exception.cli_exception import CliException, ParsingError


def check_details_validity(mode: str):
    def decorator(func):
        @wraps(func)
        def with_checker(*args, **kwargs):
            # Get params
            cluster_name = kwargs['cluster_name']

            # Get details
            try:
                cluster_details = load_cluster_details(cluster_name=cluster_name)
            except FileNotFoundError:
                raise CliException(f"Cluster {cluster_name} is not found")

            # Check details validity
            try:
                if mode == 'grass' and cluster_details['mode'] == 'grass':
                    if cluster_details['cloud']['infra'] == 'azure':
                        pass
                    else:
                        raise ParsingError(f"Details are broken: Invalid infra: {cluster_details['cloud']['infra']}")
                elif mode == 'k8s' and cluster_details['mode'] == 'k8s':
                    if cluster_details['cloud']['infra'] == 'azure':
                        pass
                    else:
                        raise ParsingError(f"Details are broken: Invalid infra: {cluster_details['cloud']['infra']}")
                else:
                    raise ParsingError(f"Details are broken: Invalid mode: {cluster_details['mode']}")
            except KeyError as e:
                raise ParsingError(f"Details are broken: Missing key: '{e.args[0]}'")

            func(*args, **kwargs)

        return with_checker

    return decorator
