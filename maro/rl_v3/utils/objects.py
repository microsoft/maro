from dataclasses import dataclass

import numpy as np


@dataclass
class ActionWithAux:
    action: np.ndarray
    value: float = None
    logp: float = None


SHAPE_CHECK_FLAG = True
