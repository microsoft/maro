# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import csv


class PortOrderExporter:
    """Utils used to export full's source and target."""

    def __init__(self, enabled: bool = False):
        self._enabled = enabled
        self._orders = []

    def add(self, order):
        """Add an order to export, it will be ignored if export is disabled.

        Args:
            order (object): Order to export.
        """
        if self._enabled:
            self._orders.append(order)

    def dump(self, folder: str):
        """Dump current orders to csv.

        Args:
            folder (str): Folder to hold dump file.
        """
        if self._enabled:
            with open(f"{folder}/orders.csv", "w+", newline="") as fp:
                writer = csv.writer(fp)

                writer.writerow(
                    ["tick", "src_port_idx", "dest_port_idx", "quantity"]
                )

                for order in self._orders:
                    writer.writerow(
                        [order.tick, order.src_port_idx, order.dest_port_idx, order.quantity])

            self._orders.clear()
