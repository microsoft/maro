# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.frame import FrameBase, FrameNode

from .cluster import Cluster
from .data_center import DataCenter
from .physical_machine import PhysicalMachine
from .rack import Rack
from .region import Region
from .zone import Zone


def build_frame(snapshots_num: int, **kwargs):
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
        regions = FrameNode(Region, kwargs["region_amount"])
        zones = FrameNode(Zone, kwargs["zone_amount"])
        data_centers = FrameNode(DataCenter, kwargs["data_center_amount"])
        clusters = FrameNode(Cluster, kwargs["cluster_amount"])
        racks = FrameNode(Rack, kwargs["rack_amount"])
        pms = FrameNode(PhysicalMachine, kwargs["pm_amount"])

        def __init__(self):
            super().__init__(enable_snapshot=True, total_snapshot=snapshots_num)

    return VmSchedulingFrame()
