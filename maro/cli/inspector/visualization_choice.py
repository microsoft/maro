# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class PanelViewChoice(Enum):
    Intra_Epoch = "Intra_Epoch View"
    Inter_Epoch = "Inter_Epoch View"


class CIMIntraViewChoice(Enum):
    by_port = "by_port"
    by_snapshot = "by_snapshot"


class CitiBikeIntraViewChoice(Enum):
    by_station = "by_station"
    by_snapshot = "by_snapshot"
