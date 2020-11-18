import os

from yaml import safe_load

from maro.cli.data_pipeline.base import DataPipeline, DataTopology
from maro.cli.data_pipeline.utils import StaticParameter, download_file
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class DataCenterPipeline(DataPipeline):
    """Generate data_center data and other necessary files for the specified topology.

    The files will be generated in ~/.maro/data/data_center.
    """

    _download_file_name = "AzurePublicDatasetLinksV2.txt"

    def __init__(self, topology: str, source: str, is_temp: bool = False):
        super().__init__(scenario="data_center", topology=topology, source=source, is_temp=is_temp)

    def download(self, is_force: bool = False):
        # Download text with all urls.
        super().download(is_force=is_force)
        if os.path.exists(self._download_file):
            # Download vm_table and cpu_reading
            logger.info_green("Downloading vmtable and cpu readings.")
            with open(self._download_file, mode="r", encoding="utf-8") as urls:
                for remote_url in urls.read().splitlines():
                    file_name = remote_url.split('/')[-1]
                    if file_name.startswith("vm"):
                        vm_data_file = os.path.join(self._download_folder, file_name)
                        if (not is_force) and os.path.exists(vm_data_file):
                            logger.info_green("File already exists, skipping download.")
                        else:
                            logger.info_green(f"Downloading VM data from {remote_url} to {vm_data_file}.")
                            download_file(source=remote_url, destination=vm_data_file)
        else:
            logger.warning(f"Not found downloaded source file: {self._download_file}.")


class DataCenterTopology(DataTopology):
    def __init__(self, topology: str, source: str, is_temp=False):
        super().__init__()
        self._data_pipeline["vm_data"] = DataCenterPipeline(topology=topology, source=source, is_temp=is_temp)


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
            for topology in self._conf["vm_data"].keys():
                self.topologies[topology] = DataCenterTopology(
                    topology=topology,
                    source=self._conf["vm_data"][topology]["remote_url"],
                    is_temp=is_temp
                )
