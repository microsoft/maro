
import os
import sys
import numpy as np

from maro.simulator import Env
from maro.simulator.snapshotwrapper import SnapshotWrapper


def main():
    env = Env(
        scenario="supply_chain",
        topology="sample",
        durations=100,
        max_snapshots=100
    )

    g = SnapshotWrapper(env)

    g.step(None)

    p = g.get_node_instances("product")[0]

    print(p.maro_attributes)

    s = g.get_node_instances("storage")[2]

    print(s.product_number)

    print(p.product_id)

    r = g.get_node_instances("retailer")[0]

    print(r.children.storage.product_number.astype(np.int))

    print(p[[0, 1, 2]:'facility_id'])


if __name__ == "__main__":
    main()
