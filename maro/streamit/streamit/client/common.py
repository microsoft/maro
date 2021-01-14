

import os
import struct

from enum import IntEnum, Enum

class MessageType(IntEnum):
    Experiment = 0
    Episode = 1
    Tick = 2
    Data = 3
    BigString = 4

    Close = 10
