# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .model import FCN_model
from .config import fcn_config

def ahu_pred_model():
    return FCN_model(**fcn_config)
