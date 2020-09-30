# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from math import floor
from typing import Union

import numpy as np

from .utils import apply_noise, clip, order_init_rand


class GlobalOrderProportion:
    """Helper used to generate order proportion at specified tick range.
    """

    def parse(self, conf: dict, total_container: int, max_tick: int, start_tick: int = 0) -> np.ndarray:
        """Parse specified configuration, and generate order proportion.

        Args:
            conf (dict): Configuration to parse.
            total_container (int): Total containers in this environment.
            max_tick (int): Max tick to generate.
            start_tick (int): Start tick to generate.

        Returns:
            np.ndarray: 1-dim numpy array for specified range.
        """
        durations: int = max_tick - start_tick

        order_proportion = np.zeros(durations, dtype="i")

        # read configurations
        period: int = conf["period"]
        noise: Union[float, int] = conf["sample_noise"]
        sample_nodes: list = [(x, y) for x, y in conf["sample_nodes"]]

        # step 1: interpolate with configured sample nodes to generate proportion in period

        # check if there is 0 and max_tick - 1 node exist ,insert if not exist
        if sample_nodes[0][0] != 0:
            sample_nodes.insert(0, (0, 0))

        if sample_nodes[-1][0] != period - 1:
            sample_nodes.append((period - 1, 0))

        # our xp is period
        xp = [node[0] for node in sample_nodes]
        yp = [node[1] for node in sample_nodes]

        # distribution per period
        order_period_distribution = np.interp(
            [t for t in range(period)], xp, yp)

        # step 2: extend to specified range

        for t in range(start_tick, max_tick):
            orders = order_period_distribution[t % period]  # ratio

            # apply noise if the distribution not zero
            if orders != 0:
                if noise != 0:
                    orders = apply_noise(orders, noise, order_init_rand)

                # clip and gen order
                orders = floor(clip(0, 1, orders) * total_container)
            order_proportion[t - start_tick] = orders

        return order_proportion
