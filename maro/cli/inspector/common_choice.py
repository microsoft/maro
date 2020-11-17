from enum import Enum


class PanelViewChoice(Enum):
    Intra_Epoch = 1
    Inter_Epoch = 2

class CIMIntraViewChoice(Enum):
    by_port =1
    by_snapshot = 2

class CitiBikeIntraViewChoice(Enum):
    by_station = 1
    by_snapshot = 2
