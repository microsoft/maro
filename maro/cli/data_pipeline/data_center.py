import os

from yaml import safe_load

from maro.cli.data_pipeline.base import DataPipeline
from maro.cli.data_pipeline.utils import StaticParameter


class DataCenterPipeline(DataPipeline):
    """Generate data_center data and other necessary files for the specified topology.

    The files will be generated in ~/.maro/data/data_center.
    """

    _download_file_name = "AzurePublicDatasetLinksV2.txt"

    def __init__(self, topology: str, source: str, is_temp: bool = False):
        super().__init__("data_center", topology, source, is_temp)

    def download(self, is_force: bool = False):
        pass


class DataCenterProcess:
    """Contains all predefined data topologies of data_center scenario."""

    meta_file_name = "source_urls.yml"
    meta_root = os.path.join(StaticParameter.data_root, "data_center/meta")

    def __init__(self, is_temp: bool = False):
        self.topologies = {}
        self.meta_root = os.path.expanduser(self.meta_root)
        self._meta_path = os.path.join(self.meta_root, self.meta_file_name)

        with open(self._meta_path) as fp:
            self._conf = safe_load(fp)
