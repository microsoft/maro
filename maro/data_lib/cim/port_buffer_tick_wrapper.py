# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from math import ceil

from .entities import CimDataCollection, NoisedItem, PortSetting
from .utils import apply_noise, buffer_tick_rand


class PortBufferTickWrapper:
    """Used to generate buffer ticks when empty/full become available.

    Examples:

        ticks = data_cntr.empty_return_buffers[port_index]

    Args:
        data (CimDataCollection): Cim data collection.
        attribute_func (callable): Function to get attribute, used to switch between empty and full.
    """

    def __init__(self, data: CimDataCollection, attribute_func: callable):
        self._ports = data.ports_settings
        self._attribute_func = attribute_func

    def __getitem__(self, key):
        port_idx = key
        port: PortSetting = self._ports[port_idx]

        buffer_setting: NoisedItem = self._attribute_func(port)

        return ceil(apply_noise(buffer_setting.base, buffer_setting.noise, buffer_tick_rand))
