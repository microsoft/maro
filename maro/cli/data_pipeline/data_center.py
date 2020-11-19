import gzip
import os
import shutil

import pandas as pd
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

    _vm_table_file_name = "vmtable.csv.gz"
    _clean_vm_table_file_name = "vmtable.csv"

    _cpu_readings_file_name = "vm_cpu_readings-file-1-of-195.csv.gz"

    def __init__(self, topology: str, source: str, is_temp: bool = False):
        super().__init__(scenario="data_center", topology=topology, source=source, is_temp=is_temp)

        self._vm_table_file = os.path.join(self._download_folder, self._vm_table_file_name)

    def download(self, is_force: bool = False):
        # Download text with all urls.
        super().download(is_force=is_force)
        if os.path.exists(self._download_file):
            # Download vm_table and cpu_reading
            logger.info_green("Downloading vmtable and cpu readings.")
            with open(self._download_file, mode="r", encoding="utf-8") as urls:
                for remote_url in urls.read().splitlines():
                    file_name = remote_url.split('/')[-1]
                    if file_name.startswith("vmtable"):
                        if (not is_force) and os.path.exists(self._vm_table_file):
                            logger.info_green("File already exists, skipping download.")
                        else:
                            logger.info_green(f"Downloading VM data from {remote_url} to {self._vm_table_file}.")
                            download_file(source=remote_url, destination=self._vm_table_file)
        else:
            logger.warning(f"Not found downloaded source file: {self._download_file}.")

    def clean(self):
        """Unzip the csv file and process it for building binary file."""
        super().clean()
        logger.info_green("Cleaning VM data.")
        if os.path.exists(self._vm_table_file):
            # Unzip gz file.
            unzip_vm_table_file = os.path.join(self._clean_folder, self._clean_vm_table_file_name)
            logger.info_green("Unzip start.")
            with gzip.open(self._vm_table_file, mode="r") as f_in:
                logger.info_green(
                    f"Unzip {self._clean_vm_table_file_name} from {self._vm_table_file} to {unzip_vm_table_file}."
                )
                with open(unzip_vm_table_file, "w") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info_green("Unzip finished.")
            # Preprocess vmtable.
        else:
            logger.warning(f"Not found downloaded source file: {self._vm_table_file}.")

    def _preprocess(self, unzip_vm_table_file: str):
        headers = [
            'vmid', 'subscriptionid', 'deploymentid', 'vmcreated', 'vmdeleted', 'maxcpu', 'avgcpu', 'p95maxcpu',
            'vmcategory', 'vmcorecountbucket', 'vmmemorybucket'
        ]
        vm_table_raw = pd.read_csv(unzip_vm_table_file, header=None, index_col=False, names=headers)
        required_headers = ['vmid', 'vmcreated', 'vmdeleted', 'vmcorecountbucket', 'vmmemorybucket']
        vm_table = vm_table_raw[required_headers]
        vm_table['lifetime'] = vm_table['vmdeleted'] - ['vmcreated']


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
