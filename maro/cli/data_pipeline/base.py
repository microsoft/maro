# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import shutil
from abc import ABC

from maro.cli.data_pipeline.utils import StaticParameter, convert, download_file, generate_name_with_uuid
from maro.utils.exception.cli_exception import CommandError
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class DataPipeline(ABC):
    """Base class of data pipeline.

    Generate scenario/topology specific data for the business engine.
    General workflow:
    Step 1: Download the original data file from the source to download folder.
    Step 2: Generate the clean data in clean folder.
    Step 3: Build a binary data file in build folder.
    The folder structure is:
    ~/.maro
            /data/[scenario]/[topology]
                                    /_download original data file
                                    /_clean cleaned data file
                                    /_build bin data file and other necessory files
                            /meta meta files for data pipeline

    Args:
        scenario(str): The scenario of the data.
        topology(str): The topology of the scenario.
        source(str): The original source of data file.
        is_temp(bool): (optional) If the data file is temporary.
    """

    _download_file_name = ""

    _clean_file_name = ""

    _build_file_name = ""

    _meta_file_name = ""

    def __init__(self, scenario: str, topology: str, source: str, is_temp: bool = False):
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

    def download(self, is_force: bool, fall_back: callable = None):
        """Download the original data file.

        Args:
            is_force(bool): If forced re-download the data file.
            fall_back(callable): (optional) Fallback function to execute when download failed.
        """
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
                if fall_back is not None:
                    logger.warning_yellow(f"Calling fall_back function: {fall_back}.")
                    fall_back()
                else:
                    raise CommandError("generate", f"Download error: {e}.")

    def clean(self):
        """Clean the original data file."""
        self._new_folder_list.append(self._clean_folder)
        os.makedirs(self._clean_folder, exist_ok=True)
        self._new_folder_list.append(self._build_folder)
        os.makedirs(self._build_folder, exist_ok=True)

    def build(self):
        """Build the cleaned data file to binary data file."""
        self._new_file_list.append(self._build_file)
        if os.path.exists(self._clean_file):
            logger.info_green(f"Building binary data from {self._clean_file} to {self._build_file}.")
            convert(meta=self._build_meta_file, file=[self._clean_file], output=self._build_file)
        else:
            logger.warning_yellow(f"Not found cleaned data: {self._clean_file}.")

    def remove_file(self):
        """Remove the temporary files."""
        for new_file in self._new_file_list:
            if os.path.exists(new_file):
                os.remove(new_file)
        self._new_file_list.clear()

    def remove_folder(self):
        """Remove the temporary folders."""
        for new_folder in self._new_folder_list:
            if os.path.exists(new_folder):
                shutil.rmtree(new_folder)
        self._new_folder_list.clear()


class DataTopology(ABC):
    """Data topology manage multi data pipelines for a specified topology of a research scenario."""

    def __init__(self):
        self._data_pipeline = {}

    def get_build_folders(self) -> dict:
        """Get the build file folders of all data pipelines for the topology.

        Returns:
            dict: Dictionary of build folders, keys are data pipeline names, values
                are paths of the build folders.
        """
        ret = {}
        for pipeline in self._data_pipeline:
            ret[pipeline] = self._data_pipeline[pipeline].build_folder
        return ret

    def download(self, is_force: bool = False):
        """Download the original data files of all data pipelines.

        Args:
            is_force(bool): If forced re-download the data file.
        """
        for pipeline in self._data_pipeline:
            self._data_pipeline[pipeline].download(is_force)

    def clean(self):
        """Clean the original data files of all data pipelines."""
        for pipeline in self._data_pipeline:
            self._data_pipeline[pipeline].clean()

    def build(self):
        """Build the cleaned data files of all data pipelines to binary data file."""
        for pipeline in self._data_pipeline:
            self._data_pipeline[pipeline].build()

    def remove(self):
        """Remove the temporary files and folders of all data pipelines."""
        for pipeline in self._data_pipeline:
            self._data_pipeline[pipeline].remove_file()
        for pipeline in self._data_pipeline:
            self._data_pipeline[pipeline].remove_folder()
