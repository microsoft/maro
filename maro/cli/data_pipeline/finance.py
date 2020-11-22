import csv
import os
import random
import zipfile
from enum import Enum

import geopy.distance
import numpy as np
import pandas as pd
from yaml import safe_load

from maro.cli.data_pipeline.base import DataPipeline, DataTopology
from maro.cli.data_pipeline.utils import StaticParameter, download_file
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)

class FinancePipeline(DataPipeline):
    