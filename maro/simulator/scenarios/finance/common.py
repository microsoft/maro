from typing import List, Dict
from .abs_sub_business_engine import AbsSubBusinessEngine
from enum import Enum

class FinanceType(Enum):
    stock = "stock",
    futures = "futures"


class SubEngineAccessWrapper:
    class PropertyAccessor:
        def __init__(self, properties: dict):
            self._properties = properties

        def __getitem__(self, name: str):
            """Used to access frame/snapshotlist by name as a dictionary."""
            if name not in self._properties:
                return None
            
            return self._properties[name]

        def __getattribute__(self, name):
            """Used to access frame/snapshotlist by name as a attribute"""
            # if name in self._properties:
                # return self._properties[name]
            properties = object.__getattribute__(self, "_properties")

            return properties[name]

    """Wrapper to access frame/config/snapshotlist by name of sub-engine"""
    def __init__(self, sub_engines: List[AbsSubBusinessEngine]):
        self._engines = {}

        # convert it to name to engine dict for easy access
        for engine in sub_engines:
            self._engines[engine.name] = engine

    def get_property_access(self, property_name: str):
        properties = {name: getattr(engine, property_name) for name, engine in self._engines.items()}

        return SubEngineAccessWrapper.PropertyAccessor(properties)

    