# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import csv


class PortOrderExporter:
    def __init__(self, enabled: bool = False):
        self._enabled = enabled
        self._orders = []

    def add(self, order):
        if self._enabled:
            self._orders.append(order)

    def dump(self, folder: str):
        if self._enabled:
            with open(f"{folder}/orders.csv", "w+", newline="") as fp:
                writer = csv.writer(fp)

                writer.writerow(["tick", "src_port_idx", "dest_port_idx", "quantity"])

                for order in self._orders:
                    writer.writerow([order.tick, order.src_port_idx, order.dest_port_idx, order.quantity])

            self._orders.clear()
