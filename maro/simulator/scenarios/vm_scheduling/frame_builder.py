# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.frame import FrameBase, FrameNode

from .cluster import Cluster
from .data_center import DataCenter
from .physical_machine import PhysicalMachine
from .rack import Rack
from .region import Region
from .zone import Zone


def build_frame(
    snapshots_num: int, region_amount: int, zone_amount: int, data_center_amount: int, cluster_amount: int,
    rack_amount: int, pm_amount: int
):
    """Function to build vm_scheduling Frame.

    Args:
        snapshot_num (int): Number of in-memory snapshots.
        region_amount (int): Number of region.
        zone_amount (int): Number of zone.
        cluster_amount (int): Number of cluster.
        pm_amount (int): Number of physical machine.

    Returns:
        VmSchedulingFrame: Frame instance for vm_scheduling scenario.
    """
    class VmSchedulingFrame(FrameBase):
        regions = FrameNode(Region, region_amount)
        zones = FrameNode(Zone, zone_amount)
        data_centers = FrameNode(DataCenter, data_center_amount)
        clusters = FrameNode(Cluster, cluster_amount)
        racks = FrameNode(Rack, rack_amount)
        pms = FrameNode(PhysicalMachine, pm_amount)

        def __init__(self):
            super().__init__(enable_snapshot=True, total_snapshot=snapshots_num)

    return VmSchedulingFrame()
