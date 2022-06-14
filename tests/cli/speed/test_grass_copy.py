# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
import logging
import os
import shutil
import time
import unittest
import uuid

import yaml

from maro.cli.utils.details_reader import DetailsReader
from maro.cli.utils.params import GlobalParams, GlobalPaths
from maro.cli.utils.subprocess import Subprocess

TEST_TO_TIME = {}


def record_speed(func):
    def with_record_speed(*args, **kwargs):
        start_time = time.time_ns() / (10**9)
        func(*args, **kwargs)
        end_time = time.time_ns() / (10**9)
        print(f"{func.__name__}: {end_time - start_time} s")
        TEST_TO_TIME[func.__name__] = end_time - start_time

    return with_record_speed


@unittest.skipUnless(os.environ.get("test_with_cli", False), "Require cli prerequisites.")
class TestGrassCopy(unittest.TestCase):
    """Tests for Grass Copy.

    Tests should be executed in specific order,
    and the order in which the various tests will be run is determined by sorting the test method names with
    respect to the built-in ordering for strings.
    Therefore, we use test_X (X is a digit) as prefix to specify the order of the tests.
    Ref: https://docs.python.org/3.7/library/unittest.html#organizing-test-code
    """

    test_id = None
    cluster_name = None

    @classmethod
    def setUpClass(cls) -> None:
        # Set params
        GlobalParams.LOG_LEVEL = logging.DEBUG
        cls.test_id = uuid.uuid4().hex[:8]
        os.makedirs(os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{cls.test_id}"), exist_ok=True)
        cls.file_path = os.path.abspath(__file__)
        cls.dir_path = os.path.dirname(cls.file_path)
        cls.deployment_template_path = os.path.normpath(
            os.path.join(cls.dir_path, "../templates/test_grass_azure_create.yml"),
        )
        cls.deployment_path = os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{cls.test_id}/test_grass_azure_create.yml")
        cls.config_path = os.path.normpath(os.path.join(cls.dir_path, "../config.yml"))

        # Load config and save deployment
        with open(cls.deployment_template_path) as fr:
            deployment_details = yaml.safe_load(fr)
        with open(cls.config_path) as fr:
            config_details = yaml.safe_load(fr)
            if config_details["cloud/subscription"] and config_details["user/admin_public_key"]:
                deployment_details["cloud"]["subscription"] = config_details["cloud/subscription"]
                deployment_details["user"]["admin_public_key"] = config_details["user/admin_public_key"]
            else:
                raise Exception("Invalid config")
        with open(cls.deployment_path, "w") as fw:
            yaml.safe_dump(deployment_details, fw)

        # Get params from deployments
        cls.cluster_name = deployment_details["name"]

        # Init test files
        cls.local_big_file_path = os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{cls.test_id}/big_file")
        cls.local_small_files_path = os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{cls.test_id}/small_files")
        command = f"dd if=/dev/zero of={cls.local_big_file_path} bs=1 count=0 seek=1G"
        Subprocess.run(command=command)
        command = f"git clone git@github.com:microsoft/maro.git {cls.local_small_files_path}"
        Subprocess.run(command=command)

        # Create cluster
        command = f"maro grass create --debug {cls.deployment_path}"
        Subprocess.interactive_run(command=command)
        cluster_details = DetailsReader.load_cluster_details(cluster_name=cls.cluster_name)
        master_details = cls._get_master_details()
        cls.admin_username = cluster_details["user"]["admin_username"]
        cls.master_public_ip_address = master_details["public_ip_address"]

    @classmethod
    def tearDownClass(cls) -> None:
        # Delete cluster
        command = f"maro grass delete --debug {cls.cluster_name}"
        Subprocess.interactive_run(command=command)

        # Print result
        print(
            json.dumps(
                TEST_TO_TIME,
                indent=4,
                sort_keys=True,
            ),
        )

        # Delete tmp test folder
        shutil.rmtree(os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{cls.test_id}"))

    # Utils

    @classmethod
    def _get_master_details(cls) -> dict:
        command = f"maro grass status {cls.cluster_name} master"
        return_str = Subprocess.run(command=command)
        return json.loads(return_str)

    # Tests

    @record_speed
    def test_1_rsync_big_file_to_remote(self) -> None:
        command = (
            f"ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'mkdir -p ~/test/{self.test_id}/test_1_rsync_big_file_to_remote'"
        )
        _ = Subprocess.run(command=command)
        command = (
            f"rsync -e 'ssh -o StrictHostKeyChecking=no' -az -r "
            f"{self.local_big_file_path} "
            f"{self.admin_username}@{self.master_public_ip_address}:"
            f"~/test/{self.test_id}/test_1_rsync_big_file_to_remote"
        )
        Subprocess.interactive_run(command=command)

    @record_speed
    def test_1_rsync_small_files_to_remote(self) -> None:
        command = (
            f"ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'mkdir -p ~/test/{self.test_id}/test_1_rsync_small_files_to_remote'"
        )
        _ = Subprocess.run(command=command)
        command = (
            f"rsync -e 'ssh -o StrictHostKeyChecking=no' -az -r "
            f"{self.local_small_files_path} "
            f"{self.admin_username}@{self.master_public_ip_address}:"
            f"~/test/{self.test_id}/test_1_rsync_small_files_to_remote"
        )
        Subprocess.interactive_run(command=command)

    @record_speed
    def test_2_rsync_big_file_to_local(self) -> None:
        command = f"mkdir -p {GlobalPaths.MARO_TEST}/{self.test_id}/test_2_rsync_big_file_to_local"
        _ = Subprocess.run(command=command)
        command = (
            f"rsync -e 'ssh -o StrictHostKeyChecking=no' -az -r "
            f"{self.admin_username}@{self.master_public_ip_address}:"
            f"~/test/{self.test_id}/test_1_rsync_big_file_to_remote "
            f"{GlobalPaths.MARO_TEST}/{self.test_id}/test_2_rsync_big_file_to_local"
        )
        Subprocess.interactive_run(command=command)
        self.assertTrue(
            os.path.exists(
                os.path.expanduser(
                    f"{GlobalPaths.MARO_TEST}/{self.test_id}/"
                    f"test_2_rsync_big_file_to_local/test_1_rsync_big_file_to_remote/big_file",
                ),
            ),
        )

    @record_speed
    def test_2_rsync_small_files_to_local(self) -> None:
        command = f"mkdir -p {GlobalPaths.MARO_TEST}/{self.test_id}/test_2_rsync_small_files_to_local"
        _ = Subprocess.run(command=command)
        command = (
            f"rsync -e 'ssh -o StrictHostKeyChecking=no' -az -r "
            f"{self.admin_username}@{self.master_public_ip_address}:"
            f"~/test/{self.test_id}/test_1_rsync_small_files_to_remote "
            f"{GlobalPaths.MARO_TEST}/{self.test_id}/test_2_rsync_small_files_to_local"
        )
        Subprocess.interactive_run(command=command)
        self.assertTrue(
            os.path.exists(
                os.path.expanduser(
                    f"{GlobalPaths.MARO_TEST}/{self.test_id}/"
                    f"test_2_rsync_small_files_to_local/test_1_rsync_small_files_to_remote/small_files/README.md",
                ),
            ),
        )

    @record_speed
    def test_1_tar_ssh_big_file_to_remote(self) -> None:
        command = (
            f"ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'mkdir -p ~/test/{self.test_id}/test_1_tar_ssh_big_file_to_remote'"
        )
        _ = Subprocess.run(command=command)

        basename = os.path.basename(self.local_big_file_path)
        dirname = os.path.dirname(self.local_big_file_path)
        command = (
            f"tar cf - -C {dirname} {basename} | "
            f"ssh {self.admin_username}@{self.master_public_ip_address} "
            f"'tar xf - -C ~/test/{self.test_id}/test_1_tar_ssh_big_file_to_remote'"
        )
        Subprocess.interactive_run(command=command)

    @record_speed
    def test_1_tar_ssh_small_files_to_remote(self) -> None:
        command = (
            f"ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'mkdir -p ~/test/{self.test_id}/test_1_tar_ssh_small_files_to_remote'"
        )
        _ = Subprocess.run(command=command)

        basename = os.path.basename(self.local_small_files_path)
        dirname = os.path.dirname(self.local_small_files_path)
        command = (
            f"tar cf - -C {dirname} {basename} | "
            f"ssh {self.admin_username}@{self.master_public_ip_address} "
            f"'tar xf - -C ~/test/{self.test_id}/test_1_tar_ssh_small_files_to_remote'"
        )
        Subprocess.interactive_run(command=command)

    @record_speed
    def test_2_tar_ssh_big_file_to_local(self) -> None:
        command = f"mkdir -p {GlobalPaths.MARO_TEST}/{self.test_id}/test_2_tar_ssh_big_file_to_local"
        _ = Subprocess.run(command=command)

        basename = os.path.basename(f"~/test/{self.test_id}/test_1_tar_ssh_big_file_to_remote")
        dirname = os.path.dirname(f"~/test/{self.test_id}/test_1_tar_ssh_big_file_to_remote")
        command = (
            f"ssh {self.admin_username}@{self.master_public_ip_address} 'tar cf - -C {dirname} {basename}' | "
            f"tar xf - -C {GlobalPaths.MARO_TEST}/{self.test_id}/test_2_tar_ssh_big_file_to_local"
        )
        Subprocess.interactive_run(command=command)
        self.assertTrue(
            os.path.exists(
                os.path.expanduser(
                    f"{GlobalPaths.MARO_TEST}/{self.test_id}/"
                    f"test_2_tar_ssh_big_file_to_local/test_1_tar_ssh_big_file_to_remote/big_file",
                ),
            ),
        )

    @record_speed
    def test_2_tar_ssh_small_files_to_local(self) -> None:
        command = f"mkdir -p {GlobalPaths.MARO_TEST}/{self.test_id}/test_2_tar_ssh_small_files_to_local"
        _ = Subprocess.run(command=command)
        basename = os.path.basename(f"~/test/{self.test_id}/test_1_tar_ssh_small_files_to_remote")
        dirname = os.path.dirname(f"~/test/{self.test_id}/test_1_tar_ssh_small_files_to_remote")
        command = (
            f"ssh {self.admin_username}@{self.master_public_ip_address} 'tar cf - -C {dirname} {basename}' | "
            f"tar xf - -C {GlobalPaths.MARO_TEST}/{self.test_id}/test_2_tar_ssh_small_files_to_local"
        )
        Subprocess.interactive_run(command=command)
        self.assertTrue(
            os.path.exists(
                os.path.expanduser(
                    f"{GlobalPaths.MARO_TEST}/{self.test_id}/"
                    f"test_2_tar_ssh_small_files_to_local/test_1_tar_ssh_small_files_to_remote/small_files/README.md",
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
