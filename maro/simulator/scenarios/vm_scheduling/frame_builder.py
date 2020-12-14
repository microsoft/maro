# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.frame import FrameBase, FrameNode

from .physical_machine import PhysicalMachine


def build_frame(pm_amount: int, snapshots_num: int):
    """Function to build vm_scheduling Frame.

    Args:
        pm_amount (int): Number of physical machine.
        snapshot_num (int): Number of in-memory snapshots.

    Returns:
        VmSchedulingFrame: Frame instance for vm_scheduling scenario.
    """
    class VmSchedulingFrame(FrameBase):
        pms = FrameNode(PhysicalMachine, pm_amount)

        def __init__(self):
            super().__init__(enable_snapshot=True, total_snapshot=snapshots_num)

    return VmSchedulingFrame()
