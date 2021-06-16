
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

    p = g.nodes["product"][0]

    s = g.nodes["storage"][2]

    print(s.product_number)

    print(p.product_id)

    r = g.nodes["retailer"][0]

    print(r.children.storage.product_number.astype(np.int))

    print(p[[0, 1, 2]:'facility_id'])

    downsteams = g.relations['downstreams']

    print(downsteams.down_to_up())


if __name__ == "__main__":
    main()
