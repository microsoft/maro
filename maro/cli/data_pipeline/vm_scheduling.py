import gzip
import os
import shutil
import time
from csv import reader, writer
from typing import List

import aria2p
import pandas as pd
from yaml import safe_load

from maro.cli.data_pipeline.base import DataPipeline, DataTopology
from maro.cli.data_pipeline.utils import StaticParameter, convert, download_file
from maro.utils.exception.cli_exception import CommandError
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class VmSchedulingPipeline(DataPipeline):
    """Generate vm_scheduling Pipeline data and other necessary files for the specified topology.

    The files will be generated in ~/.maro/data/vm_scheduling.
    """

    _download_file_name = "AzurePublicDatasetLinksV2.txt"

    _vm_table_file_name = "vmtable.csv.gz"
    _raw_vm_table_file_name = "vmtable_raw.csv"

    _clean_file_name = "vmtable.csv"
    _build_file_name = "vmtable.bin"

    _meta_file_name = "vmtable.yml"
    # VM category includes three types, converting to 0, 1, 2.
    _category_map = {'Delay-insensitive': 0, 'Interactive': 1, 'Unknown': 2}

    def __init__(self, topology: str, source: str, sample: int, seed: int, is_temp: bool = False):
        super().__init__(scenario="vm_scheduling", topology=topology, source=source, is_temp=is_temp)

        self._sample = sample
        self._seed = seed

        self._download_folder = os.path.join(self._data_root, self._scenario, ".source", ".download")
        self._raw_folder = os.path.join(self._data_root, self._scenario, ".source", ".raw")

        self._download_file = os.path.join(self._download_folder, self._download_file_name)

        self._vm_table_file = os.path.join(self._download_folder, self._vm_table_file_name)
        self._raw_vm_table_file = os.path.join(self._raw_folder, self._raw_vm_table_file_name)

        self._cpu_readings_file_name_list = []
        self._clean_cpu_readings_file_name_list = []

        self.aria2 = aria2p.API(
            aria2p.Client(
                host="http://localhost",
                port=6800,
                secret=""
            )
        )
        self._download_file_list = []

    def download(self, is_force: bool = False):
        self._new_folder_list.append(self._download_folder)
        os.makedirs(self._download_folder, exist_ok=True)

        self._new_file_list.append(self._download_file)

        if (not is_force) and os.path.exists(self._download_file):
            logger.info_green("File already exists, skipping download.")
        else:
            logger.info_green(f"Downloading data from {self._source} to {self._download_file}.")
            try:
                download_file(source=self._source, destination=self._download_file)
            except Exception as e:
                logger.warning_yellow(f"Failed to download from {self._source} to {self._download_file}.")
                raise CommandError("generate", f"Download error: {e}.")

        # Download text with all urls.
        if os.path.exists(self._download_file):
            # Download vm_table and cpu_readings
            self._aria2p_download(is_force=is_force)
        else:
            logger.warning(f"Not found downloaded source file: {self._download_file}.")

    def _aria2p_download(self, is_force: bool = False) -> List[list]:
        """Read from the text file which contains urls and use aria2p to download.

        Args:
            is_force (bool): Is force or not.
        """
        logger.info_green("Downloading vmtable and cpu readings.")
        # Download parts of cpu reading files.
        num_files = 195
        # Open the txt file which contains all the required urls.
        with open(self._download_file, mode="r", encoding="utf-8") as urls:
            for remote_url in urls.read().splitlines():
                # Get the file name.
                file_name = remote_url.split('/')[-1]
                # Two kinds of required files "vmtable" and "vm_cpu_readings-" start with vm.
                if file_name.startswith("vmtable"):
                    if (not is_force) and os.path.exists(self._vm_table_file):
                        logger.info_green(f"{self._vm_table_file} already exists, skipping download.")
                    else:
                        logger.info_green(f"Downloading vmtable from {remote_url} to {self._vm_table_file}.")
                        self.aria2.add_uris(uris=[remote_url], options={'dir': self._download_folder})

                elif file_name.startswith("vm_cpu_readings") and num_files > 0:
                    num_files -= 1
                    cpu_readings_file = os.path.join(self._download_folder, file_name)
                    self._cpu_readings_file_name_list.append(file_name)

                    if (not is_force) and os.path.exists(cpu_readings_file):
                        logger.info_green(f"{cpu_readings_file} already exists, skipping download.")
                    else:
                        logger.info_green(f"Downloading cpu_readings from {remote_url} to {cpu_readings_file}.")
                        self.aria2.add_uris(uris=[remote_url], options={'dir': f"{self._download_folder}"})

        self._check_all_download_completed()

    def _check_all_download_completed(self):
        """Check all download tasks are completed and remove the ".aria2" files."""

        while 1:
            downloads = self.aria2.get_downloads()
            if len(downloads) == 0:
                logger.info_green("Doesn't exist any pending file.")
                break

            if all([download.is_complete for download in downloads]):
                # Remove temp .aria2 files.
                self.aria2.remove(downloads)
                logger.info_green("Download finished.")
                break

            for download in downloads:
                logger.info_green(f"{download.name}, {download.status}, {download.progress:.2f}%")
            logger.info_green("-" * 60)
            time.sleep(10)

    def _unzip_file(self, original_file_name: str, raw_file_name: str):
        original_file = os.path.join(self._download_folder, original_file_name)
        if os.path.exists(original_file):
            raw_file = os.path.join(self._raw_folder, raw_file_name)
            if os.path.exists(raw_file):
                logger.info_green(f"{raw_file} already exists, skipping unzip.")
            else:
                # Unzip gz file.
                with gzip.open(original_file, mode="rb") as f_in:
                    logger.info_green(
                        f"Unzip {original_file} to {raw_file}."
                    )
                    with open(raw_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
        else:
            logger.warning(f"Not found downloaded source file: {original_file}.")

    def clean(self):
        """Unzip the csv file and process it for building binary file."""
        super().clean()
        self._new_folder_list.append(self._raw_folder)
        os.makedirs(self._raw_folder, exist_ok=True)

        logger.info_green("Cleaning VM data.")
        # Unzip vmtable.
        self._unzip_file(original_file_name=self._vm_table_file_name, raw_file_name=self._raw_vm_table_file_name)
        # Unzip cpu readings.
        for cpu_readings_file_name in self._cpu_readings_file_name_list:
            raw_file_name = cpu_readings_file_name.split(".")[0] + "_raw.csv"
            self._clean_cpu_readings_file_name_list.append(cpu_readings_file_name[:-3])
            self._unzip_file(original_file_name=cpu_readings_file_name, raw_file_name=raw_file_name)
        # Preprocess.
        self._preprocess()

    def _generate_id_map(self, old_id):
        num = len(old_id)
        new_id_list = [i for i in range(1, num + 1)]
        id_map = dict(zip(old_id, new_id_list))

        return id_map

    def _process_vm_table(self, raw_vm_table_file: str):
        """Process vmtable file."""

        headers = [
            'vmid', 'subscriptionid', 'deploymentid', 'vmcreated', 'vmdeleted', 'maxcpu', 'avgcpu', 'p95maxcpu',
            'vmcategory', 'vmcorecountbucket', 'vmmemorybucket'
        ]

        required_headers = [
            'vmid', 'subscriptionid', 'deploymentid', 'vmcreated', 'vmdeleted', 'vmcategory',
            'vmcorecountbucket', 'vmmemorybucket'
        ]

        vm_table = pd.read_csv(raw_vm_table_file, header=None, index_col=False, names=headers)
        vm_table = vm_table.loc[:, required_headers]
        # Convert to tick by dividing by 300 (5 minutes).
        vm_table['vmcreated'] = pd.to_numeric(vm_table['vmcreated'], errors="coerce", downcast="integer") // 300
        vm_table['vmdeleted'] = pd.to_numeric(vm_table['vmdeleted'], errors="coerce", downcast="integer") // 300
        # The lifetime of the VM is deleted time - created time + 1 (tick).
        vm_table['lifetime'] = vm_table['vmdeleted'] - vm_table['vmcreated'] + 1

        vm_table['vmcategory'] = vm_table['vmcategory'].map(self._category_map)

        # Transform vmcorecount '>24' bucket to 32 and vmmemory '>64' to 128.
        vm_table = vm_table.replace({'vmcorecountbucket': '>24'}, 32)
        vm_table = vm_table.replace({'vmmemorybucket': '>64'}, 128)
        vm_table['vmcorecountbucket'] = pd.to_numeric(
            vm_table['vmcorecountbucket'], errors="coerce", downcast="integer"
        )
        vm_table['vmmemorybucket'] = pd.to_numeric(vm_table['vmmemorybucket'], errors="coerce", downcast="integer")
        vm_table.dropna(inplace=True)

        vm_table = vm_table.sort_values(by='vmcreated', ascending=True)

        # Generate ID map.
        vm_id_map = self._generate_id_map(vm_table['vmid'].unique())
        sub_id_map = self._generate_id_map(vm_table['subscriptionid'].unique())
        deployment_id_map = self._generate_id_map(vm_table['deploymentid'].unique())

        id_maps = (vm_id_map, sub_id_map, deployment_id_map)

        # Mapping IDs.
        vm_table['vmid'] = vm_table['vmid'].map(vm_id_map)
        vm_table['subscriptionid'] = vm_table['subscriptionid'].map(sub_id_map)
        vm_table['deploymentid'] = vm_table['deploymentid'].map(deployment_id_map)

        # Sampling the VM table.
        # 2695548 is the total number of vms in the original Azure public dataset.
        if self._sample < 2695548:
            vm_table = vm_table.sample(n=self._sample, random_state=self._seed)
            vm_table = vm_table.sort_values(by='vmcreated', ascending=True)

        return id_maps, vm_table

    def _convert_cpu_readings_id(self, old_data_path: str, new_data_path: str, vm_id_map: dict):
        """Convert vmid in each cpu readings file."""
        with open(old_data_path, 'r') as f_in:
            csv_reader = reader(f_in)
            with open(new_data_path, 'w') as f_out:
                csv_writer = writer(f_out)
                csv_writer.writerow(['timestamp', 'vmid', 'maxcpu'])
                for row in csv_reader:
                    # [timestamp, vmid, mincpu, maxcpu, avgcpu]
                    if row[1] in vm_id_map:
                        new_row = [int(row[0]) // 300, vm_id_map[row[1]], row[3]]
                        csv_writer.writerow(new_row)

    def _write_id_map_to_csv(self, id_maps):
        file_name = ['vm_id_map', 'sub_id_map', 'deployment_id_map']
        for index in range(len(id_maps)):
            id_map = id_maps[index]
            with open(os.path.join(self._raw_folder, file_name[index]) + '.csv', 'w') as f:
                csv_writer = writer(f)
                csv_writer.writerow(['original_id', 'new_id'])
                for key, value in id_map.items():
                    csv_writer.writerow([key, value])

    def _filter_out_vmid(self, vm_table: pd.DataFrame, vm_id_map: dict) -> dict:
        new_id_map = {}
        for key, value in vm_id_map.items():
            if value in vm_table.vmid.values:
                new_id_map[key] = value

        return new_id_map

    def _preprocess(self):
        logger.info_green("Process vmtable data.")
        # Process vmtable file.
        id_maps, vm_table = self._process_vm_table(raw_vm_table_file=self._raw_vm_table_file)
        filtered_vm_id_map = self._filter_out_vmid(vm_table=vm_table, vm_id_map=id_maps[0])

        with open(self._clean_file, mode="w", encoding="utf-8", newline="") as f:
            vm_table.to_csv(f, index=False, header=True)

        logger.info_green("Writing id maps file.")
        self._write_id_map_to_csv(id_maps=id_maps)

        logger.info_green("Reading cpu data.")
        # Process every cpu readings file.
        for clean_cpu_readings_file_name in self._clean_cpu_readings_file_name_list:
            raw_cpu_readings_file_name = clean_cpu_readings_file_name.split(".")[0] + "_raw.csv"
            raw_cpu_readings_file = os.path.join(self._raw_folder, raw_cpu_readings_file_name)
            clean_cpu_readings_file = os.path.join(self._clean_folder, clean_cpu_readings_file_name)
            # Convert vmid.
            logger.info_green(f"Process {clean_cpu_readings_file}.")
            self._convert_cpu_readings_id(
                old_data_path=raw_cpu_readings_file,
                new_data_path=clean_cpu_readings_file,
                vm_id_map=filtered_vm_id_map
            )

    def build(self):
        super().build()
        for clean_cpu_readings_file_name in self._clean_cpu_readings_file_name_list:
            clean_cpu_readings_file = os.path.join(self._clean_folder, clean_cpu_readings_file_name)
            if os.path.exists(clean_cpu_readings_file):
                build_file_name = clean_cpu_readings_file_name.split(".")[0] + ".bin"
                build_file = os.path.join(self._build_folder, build_file_name)
                logger.info_green(f"Building binary data from {clean_cpu_readings_file} to {build_file}.")
                cpu_meta_file = os.path.join(self._meta_folder, "cpu_readings.yml")
                convert(meta=cpu_meta_file, file=[clean_cpu_readings_file], output=build_file)
            else:
                logger.warning_yellow(f"Not found cleaned data: {self._clean_file}.")


class VmSchedulingTopology(DataTopology):
    def __init__(self, topology: str, source: str, sample: int, seed: int, is_temp=False):
        super().__init__()
        self._data_pipeline["vm_data"] = VmSchedulingPipeline(
            topology=topology,
            source=source,
            sample=sample,
            seed=seed,
            is_temp=is_temp
        )


class VmSchedulingProcess:
    """Contains all predefined data topologies of vm_scheduling scenario."""

    meta_file_name = "source_urls.yml"
    meta_root = os.path.join(StaticParameter.data_root, "vm_scheduling/meta")

    def __init__(self, is_temp: bool = False):
        self.topologies = {}
        self.meta_root = os.path.expanduser(self.meta_root)
        self._meta_path = os.path.join(self.meta_root, self.meta_file_name)

        with open(self._meta_path) as fp:
            self._conf = safe_load(fp)
            for topology in self._conf["vm_data"].keys():
                self.topologies[topology] = VmSchedulingTopology(
                    topology=topology,
                    source=self._conf["vm_data"][topology]["remote_url"],
                    sample=self._conf["vm_data"][topology]["sample"],
                    seed=self._conf["vm_data"][topology]["seed"],
                    is_temp=is_temp
                )
