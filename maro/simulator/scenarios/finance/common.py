
from enum import Enum

class FinanceType(Enum):
    Stock = "stock",
    Futures = "futures"

class SubEngineAccessWrapper:
    """Wrapper to access frame/config/snapshotlist"""
    def __init__(self):
        pass