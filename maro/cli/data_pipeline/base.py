import os
import shutil
import tempfile

from abc import ABC, abstractmethod

from maro.cli.data_pipeline.utils import convert, download_file, StaticParameter, generate_name_with_uuid
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)

class DataPipeline(ABC):
    _download_file_name = ""

    _clean_file_name = ""

    _build_file_name = ""

    _meta_file_name = ""

    def __init__(self, scenario: str, topology: str, source: str, is_temp: bool = False):
        """
        Base class of data pipeline.
        Generate scenario/topology specific data for the business engine.
        General workflow:
        Step 1: Download the original data file from the source to download folder
        Step 2: Generate the clean data in clean folder
        Step 3: Build a binary data file in build folder.
        The folder structer is:
        ~/.maro
                /data/[scenario]/[topology]
                                        /_download original data file
                                        /_clean cleaned data file
                                        /_build bin data file and other necessory files
                                /meta meta files for data pipeline


        Args:
            scenario(str): the scenario of the data 
            topology(str): the topology of the scenario
            source(str): the original source of data file
            is_temp(bool): (optional) if the data file is temporary
        """
        self._scenario = scenario
        self._topology = topology
        self._is_temp = is_temp
        self._source = source
        self._data_root = StaticParameter.data_root
        self._meta_folder = os.path.join(StaticParameter.data_root, scenario, "meta")
        self._build_folder = os.path.join(self._data_root, self._scenario, ".build", self._topology)
        self._clean_folder = os.path.join(self._data_root, self._scenario, ".source", ".clean", self._topology)
        self._download_folder = os.path.join(self._data_root, self._scenario, ".source", ".download", self._topology)

        self._new_file_list = []
        self._new_folder_list = []

        if self._is_temp:
            tmpdir = generate_name_with_uuid()
            self._download_folder = os.path.join(self._download_folder, tmpdir)
            self._clean_folder = os.path.join(self._clean_folder, tmpdir)
            self._build_folder = os.path.join(self._build_folder, tmpdir)
            
            

        self._download_file = os.path.join(self._download_folder, self._download_file_name)
        self._clean_file = os.path.join(self._clean_folder, self._clean_file_name)
        self._build_file = os.path.join(self._build_folder, self._build_file_name)

        self._build_meta_file = os.path.join(self._meta_folder, self._meta_file_name)

    @property
    def build_folder(self):
        return self._build_folder

    def download(self, is_force: bool):
        """download the original data file"""
        self._new_folder_list.append(self._download_folder)
        os.makedirs(self._download_folder, exist_ok=True)

        self._new_file_list.append(self._download_file)

        if (not is_force) and os.path.exists(self._download_file):
            logger.info_green("File already exists, skipping download.")
        else:
            logger.info_green(f"Downloading data from {self._source} to {self._download_file}")
            download_file(source=self._source, destination=self._download_file)


    def clean(self):
        """clean the original data file"""
        self._new_folder_list.append(self._clean_folder)
        os.makedirs(self._clean_folder, exist_ok=True)
        self._new_folder_list.append(self._build_folder)
        os.makedirs(self._build_folder, exist_ok=True)

    def build(self):
        """build the cleaned data file to binary data file"""            
        self._new_file_list.append(self._build_file)
        if os.path.exists(self._clean_file):
            logger.info_green(f"Building binary data from {self._clean_file} to {self._build_file}")
            convert(meta=self._build_meta_file, file=[self._clean_file], output=self._build_file)
        else:
            logger.info_green(f"Not found cleaned data: {self._clean_file}")

    def remove_file(self):
        """remove the temporary files"""
        for new_file in self._new_file_list:
            if os.path.exists(new_file):
                os.remove(new_file)
        self._new_file_list.clear()

    def remove_folder(self):
        """remove the temporary folders"""
        for new_folder in self._new_folder_list:
            if os.path.exists(new_folder):
                shutil.rmtree(new_folder)
        self._new_folder_list.clear()


class DataTopology(ABC):

    def __init__(self):
        """
        Data topology manage multi data pipelines for a specified topology of a research scenario.
        """
        self._data_pipeline = {}

    def get_build_folders(self):
        ret = {}
        for pipeline in self._data_pipeline:
            ret[pipeline] = self._data_pipeline[pipeline].build_folder
        return ret

    def download(self, is_force: bool = False):
        for pipeline in self._data_pipeline:
            self._data_pipeline[pipeline].download(is_force)

    def clean(self):
        for pipeline in self._data_pipeline:
            self._data_pipeline[pipeline].clean()

    def build(self):
        for pipeline in self._data_pipeline:
            self._data_pipeline[pipeline].build()

    def remove(self):
        for pipeline in self._data_pipeline:
            self._data_pipeline[pipeline].remove_file()
        for pipeline in self._data_pipeline:
            self._data_pipeline[pipeline].remove_folder()