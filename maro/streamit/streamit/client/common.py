

import os
import struct

from enum import IntEnum, Enum

class MessageType(IntEnum):
    BeginExperiment = 0
    BeginEpisode = 2
    BeginTick = 4
    Data = 6
    Category = 7

class DataType(Enum):
    Bool = "boolean"
    Byte = "byte"
    Short = "short"
    Char = "char"
    Int = "int"
    Float = "float"
    String = "string"
    Long = "long"
    Date = "date"
    Timestamp = "timestamp"
    Double = "double"
    Binary = "binary"
    Long256 = "long256"