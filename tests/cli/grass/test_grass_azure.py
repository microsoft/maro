# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
import logging
import os
import platform
import shutil
import time
import unittest
import uuid

import yaml

from maro.cli.grass.utils.params import NodeStatus
from maro.cli.utils.azure_controller import AzureController
from maro.cli.utils.params import GlobalParams, GlobalPaths
from maro.cli.utils.subprocess import Subprocess
from maro.utils.exception.cli_exception import CommandExecutionError

from tests.cli.utils import record_running_time


@unittest.skipUnless(os.environ.get("test_with_cli", False), "Require CLI prerequisites.")
class TestGrassAzure(unittest.TestCase):
    """Tests for Grass/Azure Mode.

    Tests should be executed in specific order,
    and the order in which the various tests will be run is determined by sorting the test method names with
    respect to the built-in ordering for strings.
    We use testXX (X is a digit) as prefix to specify the order of the tests.
    Ref: https://docs.python.org/3.7/library/unittest.html#organizing-test-code
    """

    # Tests related.
    test_id = uuid.uuid4().hex[:8]
    test_name = "test_job"
    test_func_to_time = {}

    # Paths related.
    test_file_path = os.path.abspath(__file__)
    test_dir_path = os.path.dirname(test_file_path)
    maro_pkg_path = os.path.normpath(os.path.join(test_file_path, "../../../../"))
    test_config_path = os.path.normpath(os.path.join(test_dir_path, "../config.yml"))
    create_deployment_template_path = os.path.normpath(
        path=os.path.join(test_dir_path, "./modes/azure/grass_azure_create.yml"),
    )
    create_deployment_path = f"{GlobalPaths.ABS_MARO_TEST}/{test_id}/grass_azure_create.yml"

    # Azure related.
    resource_group = None

    # Cluster related.
    cluster_name = None

    @classmethod
    def setUpClass(cls) -> None:
        # Set Env.
        GlobalParams.LOG_LEVEL = logging.DEBUG

        # Init folders.
        os.makedirs(f"{GlobalPaths.ABS_MARO_TEST}/{cls.test_id}", exist_ok=False)

        # Load config and save deployment.
        with open(file=cls.create_deployment_template_path, mode="r") as fr:
            create_deployment = yaml.safe_load(fr)
        with open(file=cls.test_config_path, mode="r") as fr:
            test_config = yaml.safe_load(fr)
            if test_config["cloud/subscription"] and test_config["cloud/default_public_key"]:
                create_deployment["name"] = f"test_maro_grass_{cls.test_id}"
                create_deployment["cloud"]["subscription"] = test_config["cloud/subscription"]
                create_deployment["cloud"]["resource_group"] = f"test_maro_grass_{cls.test_id}"
                create_deployment["cloud"]["default_public_key"] = test_config["cloud/default_public_key"]
            else:
                raise Exception("Invalid config")
        with open(file=cls.create_deployment_path, mode="w") as fw:
            yaml.safe_dump(create_deployment, fw)

        # Get params from deployments.
        cls.resource_group = create_deployment["cloud"]["resource_group"]
        cls.cluster_name = create_deployment["name"]

        # Pull testing images.
        command = "docker pull alpine:latest"
        Subprocess.run(command=command)
        command = "docker pull ubuntu:latest"
        Subprocess.run(command=command)

    @classmethod
    def tearDownClass(cls) -> None:
        # Print result.
        print(
            json.dumps(
                cls.test_func_to_time,
                indent=4,
                sort_keys=True,
            ),
        )

        # Delete resource group.
        AzureController.delete_resource_group(resource_group=cls.resource_group)

        # Delete tmp test folder.
        shutil.rmtree(f"{GlobalPaths.ABS_MARO_TEST}/{cls.test_id}")

        # Delete docker image.
        try:
            command = "docker rmi maro_runtime_cpu:test"
            Subprocess.run(command=command)
        except CommandExecutionError:
            pass

    # Utils

    def _list_nodes_details(self) -> list:
        command = f"maro grass node list {self.cluster_name}"
        return_str = Subprocess.run(command=command)
        return json.loads(return_str)

    def _get_master_details(self) -> dict:
        command = f"maro grass status {self.cluster_name} master"
        return_str = Subprocess.run(command=command)
        return json.loads(return_str)

    def _list_jobs_details(self) -> list:
        command = f"maro grass job list {self.cluster_name}"
        return_str = Subprocess.run(command=command)
        return json.loads(return_str)

    def _get_name_to_job_details(self) -> dict:
        jobs_details = self._list_jobs_details()
        name_to_job_details = {}
        for job_details in jobs_details:
            name_to_job_details[job_details["name"]] = job_details
        return name_to_job_details

    @staticmethod
    def _gracefully_wait(secs: int = 10) -> None:
        time.sleep(secs)

    # Tests

    @record_running_time(func_to_time=test_func_to_time)
    def test10_create(self) -> None:
        # Run command.
        command = f"maro grass create --debug {self.create_deployment_path}"
        Subprocess.interactive_run(command=command)

    @record_running_time(func_to_time=test_func_to_time)
    def test11_node1(self) -> None:
        """Scale node spec Standard_D2s_v3 to 1.

        A Standard_D2s_v3 should be in running state.

        Returns:
            None.
        """
        # Run command.
        command = f"maro grass node scale {self.cluster_name} --debug Standard_D2s_v3 1"
        Subprocess.interactive_run(command=command)
        self._gracefully_wait()

        # Check validity, failed if does not meet the desired state in 120s.
        is_valid = False
        start_time = time.time()
        while not is_valid and start_time + 120 >= time.time():
            try:
                is_valid = True
                nodes_details = self._list_nodes_details()
                self.assertEqual(len(nodes_details), 1)
                for node_details in nodes_details:
                    self.assertEqual(node_details["state"]["status"], NodeStatus.RUNNING)
            except AssertionError:
                is_valid = False
                time.sleep(10)
        self.assertTrue(is_valid)

    @unittest.skipIf(os.environ.get("training_only", False), "Skip if we want to test training stage only.")
    @record_running_time(func_to_time=test_func_to_time)
    def test12_image1(self) -> None:
        """Push image alpine:latest to the cluster.

        The only Standard_D2s_v3 should have loaded the image alpine:latest.

        Returns:
            None.
        """
        # Run command.
        command = f"maro grass image push {self.cluster_name} --debug --image-name alpine:latest"
        Subprocess.interactive_run(command=command)
        self._gracefully_wait()

        # Check validity, failed if does not meet the desired state in 120s.
        is_valid = False
        start_time = time.time()
        while not is_valid and start_time + 120 >= time.time():
            try:
                is_valid = True
                nodes_details = self._list_nodes_details()
                self.assertEqual(len(nodes_details), 1)
                for node_details in nodes_details:
                    self.assertEqual(node_details["state"]["status"], NodeStatus.RUNNING)
                    self.assertIn("alpine_latest", node_details["image_files"])
            except AssertionError:
                is_valid = False
                time.sleep(10)
        self.assertTrue(is_valid)

    @unittest.skipIf(os.environ.get("training_only", False), "Skip if we want to test training stage only.")
    @record_running_time(func_to_time=test_func_to_time)
    def test13_node2(self) -> None:
        """Scale node spec Standard_D2s_v3 to 2.

        Two Standard_D2s_v3 should be in running state, and they should have loaded the image alpine:latest.

        Returns:
            None.
        """
        # Run command.
        command = f"maro grass node scale {self.cluster_name} --debug Standard_D2s_v3 2"
        Subprocess.interactive_run(command=command)
        self._gracefully_wait()

        # Check validity, failed if does not meet the desired state in 120s.
        is_valid = False
        start_time = time.time()
        while not is_valid and start_time + 120 >= time.time():
            try:
                is_valid = True
                nodes_details = self._list_nodes_details()
                self.assertEqual(len(nodes_details), 2)
                for node_details in nodes_details:
                    self.assertEqual(node_details["state"]["status"], NodeStatus.RUNNING)
                    self.assertIn("alpine_latest", node_details["image_files"])
            except AssertionError:
                is_valid = False
                time.sleep(10)
        self.assertTrue(is_valid)

    @unittest.skipIf(os.environ.get("training_only", False), "Skip if we want to test training stage only.")
    @record_running_time(func_to_time=test_func_to_time)
    def test14_stop(self) -> None:
        """Stop one Standard_D2s_v3.

        One Standard_D2s_v3 should be in running state, and the other should be in Stopped state.

        Returns:
            None.
        """
        # Run command.
        command = f"maro grass node stop {self.cluster_name} --debug Standard_D2s_v3 1"
        Subprocess.interactive_run(command=command)
        self._gracefully_wait()

        # Check validity, failed if does not meet the desired state in 120s.
        is_valid = False
        start_time = time.time()
        while not is_valid and start_time + 120 >= time.time():
            try:
                is_valid = True
                nodes_details = self._list_nodes_details()
                self.assertEqual(len(nodes_details), 2)
                running_count = 0
                stopped_count = 0
                for node_details in nodes_details:
                    if node_details["state"]["status"] == NodeStatus.RUNNING:
                        running_count += 1
                    if node_details["state"]["status"] == NodeStatus.STOPPED:
                        stopped_count += 1
                self.assertEqual(running_count, 1)
                self.assertEqual(stopped_count, 1)
            except AssertionError:

                is_valid = False
                time.sleep(10)
        self.assertTrue(is_valid)

    @unittest.skipIf(os.environ.get("training_only", False), "Skip if we want to test training stage only.")
    @record_running_time(func_to_time=test_func_to_time)
    def test15_image2(self) -> None:
        """Push image ubuntu:latest to the cluster.

        The only Running Standard_D2s_v3 should have loaded the image ubuntu:latest,
        and the other should have not.

        Returns:
            None.
        """
        # Run command.
        command = f"maro grass image push {self.cluster_name} --debug --image-name ubuntu:latest"
        Subprocess.interactive_run(command=command)
        self._gracefully_wait()

        # Check validity, failed if does not meet the desired state in 120s.
        is_valid = False
        start_time = time.time()
        while not is_valid and start_time + 120 >= time.time():
            try:
                is_valid = True
                nodes_details = self._list_nodes_details()
                self.assertEqual(len(nodes_details), 2)
                running_count = 0
                stopped_count = 0
                for node_details in nodes_details:
                    if node_details["state"]["status"] == NodeStatus.RUNNING:
                        running_count += 1
                        self.assertIn("alpine_latest", node_details["image_files"])
                        self.assertIn("ubuntu_latest", node_details["image_files"])
                    if node_details["state"]["status"] == NodeStatus.STOPPED:
                        stopped_count += 1
                        self.assertIn("alpine_latest", node_details["image_files"])
                        self.assertNotIn("ubuntu_latest", node_details["image_files"])
                self.assertEqual(running_count, 1)
                self.assertEqual(stopped_count, 1)
            except AssertionError:
                is_valid = False
                time.sleep(10)
        self.assertTrue(is_valid)

    @unittest.skipIf(os.environ.get("training_only", False), "Skip if we want to test training stage only.")
    @record_running_time(func_to_time=test_func_to_time)
    def test16_start(self) -> None:
        """Start one Standard_D2s_v3.

        Two Standard_D2s_v3 should be in running state,
        and they should have loaded the image alpine:latest and ubuntu:latest.

        Returns:
            None.
        """
        # Run command.
        command = f"maro grass node start {self.cluster_name} --debug Standard_D2s_v3 1"
        Subprocess.interactive_run(command=command)
        self._gracefully_wait()

        # Check validity, failed if does not meet the desired state in 120s.
        is_valid = False
        start_time = time.time()
        while not is_valid and start_time + 120 >= time.time():
            try:
                is_valid = True
                nodes_details = self._list_nodes_details()
                self.assertEqual(len(nodes_details), 2)
                for node_details in nodes_details:
                    self.assertEqual(node_details["state"]["status"], NodeStatus.RUNNING)
                    self.assertIn("alpine_latest", node_details["image_files"])
                    self.assertIn("ubuntu_latest", node_details["image_files"])
            except AssertionError:
                is_valid = False
                time.sleep(10)
        self.assertTrue(is_valid)

    @unittest.skipIf(os.environ.get("training_only", False), "Skip if we want to test training stage only.")
    @record_running_time(func_to_time=test_func_to_time)
    def test17_data(self) -> None:
        # Create tmp files.
        os.makedirs(f"{GlobalPaths.ABS_MARO_TEST}/{self.test_id}/push/test_data", exist_ok=True)
        os.makedirs(f"{GlobalPaths.ABS_MARO_TEST}/{self.test_id}/pull", exist_ok=True)
        if platform.system() == "Windows":
            command = f"fsutil file createnew {GlobalPaths.ABS_MARO_TEST}/{self.test_id}/push/test_data/a.file 1048576"
        else:
            command = f"fallocate -l 1M {GlobalPaths.ABS_MARO_TEST}/{self.test_id}/push/test_data/a.file"
        Subprocess.run(command=command)

        # Push file to an existed folder.
        command = (
            f"maro grass data push {self.cluster_name} --debug "
            f"'{GlobalPaths.MARO_TEST}/{self.test_id}/push/test_data/a.file' '/'"
        )
        Subprocess.interactive_run(command=command)

        # Push file to a new folder.
        command = (
            f"maro grass data push {self.cluster_name} --debug "
            f"'{GlobalPaths.MARO_TEST}/{self.test_id}/push/test_data/a.file' '/F1'"
        )
        Subprocess.interactive_run(command=command)

        # Push folder to an existed folder.
        command = (
            f"maro grass data push {self.cluster_name} --debug "
            f"'{GlobalPaths.MARO_TEST}/{self.test_id}/push/test_data/' '/'"
        )
        Subprocess.interactive_run(command=command)

        # Push folder to a new folder.
        command = (
            f"maro grass data push {self.cluster_name} --debug "
            f"'{GlobalPaths.MARO_TEST}/{self.test_id}/push/test_data/' '/F2'"
        )
        Subprocess.interactive_run(command=command)

        # Pull file to an existed folder.
        command = (
            f"maro grass data pull {self.cluster_name} --debug "
            f"'/a.file' '{GlobalPaths.MARO_TEST}/{self.test_id}/pull'"
        )
        Subprocess.interactive_run(command=command)

        # Pull file to a new folder.
        command = (
            f"maro grass data pull {self.cluster_name} --debug "
            f"'/F1/a.file' '{GlobalPaths.MARO_TEST}/{self.test_id}/pull/F1'"
        )
        Subprocess.interactive_run(command=command)

        # Pull folder to an existed folder.
        command = (
            f"maro grass data pull {self.cluster_name} --debug "
            f"'/test_data' '{GlobalPaths.MARO_TEST}/{self.test_id}/pull'"
        )
        Subprocess.interactive_run(command=command)

        # Pull folder to a new folder.
        command = (
            f"maro grass data pull {self.cluster_name} --debug "
            f"'/F2/test_data/' '{GlobalPaths.MARO_TEST}/{self.test_id}/pull/F2/'"
        )
        Subprocess.interactive_run(command=command)

        self.assertTrue(os.path.exists(f"{GlobalPaths.ABS_MARO_TEST}/{self.test_id}/pull/a.file"))
        self.assertTrue(os.path.exists(f"{GlobalPaths.ABS_MARO_TEST}/{self.test_id}/pull/F1/a.file"))
        self.assertTrue(os.path.exists(f"{GlobalPaths.ABS_MARO_TEST}/{self.test_id}/pull/test_data"))
        self.assertTrue(os.path.exists(f"{GlobalPaths.ABS_MARO_TEST}/{self.test_id}/pull/F2/test_data"))

    @unittest.skipIf(os.environ.get("orchestration_only", False), "Skip if we want to test orchestration stage only.")
    @record_running_time(func_to_time=test_func_to_time)
    def test20_train_env_provision(self):
        # Build docker image and load docker image.
        command = (
            f"docker build -f {self.maro_pkg_path}/docker_files/cpu.runtime.source.df -t maro_runtime_cpu:test "
            f"{self.maro_pkg_path}"
        )
        Subprocess.run(command=command)

        # Run command.
        command = f"maro grass image push {self.cluster_name} --debug --image-name maro_runtime_cpu:test"
        Subprocess.interactive_run(command=command)

        # Check image status, failed if does not meet the desired state in 1000s.
        is_loaded = False
        start_time = time.time()
        while not is_loaded and start_time + 1000 >= time.time():
            try:
                is_loaded = True
                nodes_details = self._list_nodes_details()
                for node_details in nodes_details:
                    self.assertIn("maro_runtime_cpu_test", node_details["image_files"])
            except AssertionError:
                is_loaded = False
                time.sleep(10)
        self.assertTrue(is_loaded)

    @unittest.skipIf(os.environ.get("orchestration_only", False), "Skip if we want to test orchestration stage only.")
    @record_running_time(func_to_time=test_func_to_time)
    def test21_train_dqn(self) -> None:
        # Copy dqn examples to test folder.
        dqn_source_dir = os.path.normpath(os.path.join(self.maro_pkg_path, "./examples/cim/dqn"))
        dqn_target_dir = f"{GlobalPaths.ABS_MARO_TEST}/{self.test_id}/train/dqn"
        os.makedirs(os.path.dirname(dqn_target_dir), exist_ok=True)
        command = f"cp -r {dqn_source_dir} {dqn_target_dir}"
        Subprocess.run(command=command)

        # Get cluster details and rebuild config.
        master_details = self._get_master_details()
        with open(f"{dqn_target_dir}/config.yml", "r") as fr:
            config = yaml.safe_load(fr)
        with open(f"{dqn_target_dir}/distributed_config.yml", "r") as fr:
            distributed_config = yaml.safe_load(fr)
        with open(f"{dqn_target_dir}/config.yml", "w") as fw:
            config["main_loop"]["max_episode"] = 25
            config["main_loop"]["exploration"]["split_ep"] = 20
            yaml.safe_dump(config, fw)
        with open(f"{dqn_target_dir}/distributed_config.yml", "w") as fw:
            distributed_config["redis"]["hostname"] = master_details["private_ip_address"]
            distributed_config["redis"]["port"] = master_details["redis"]["port"]
            yaml.safe_dump(distributed_config, fw)

        # Push dqn folder to cluster.
        command = (
            f"maro grass data push {self.cluster_name} --debug "
            f"'{GlobalPaths.MARO_TEST}/{self.test_id}/train/dqn' '/train'"
        )
        Subprocess.run(command=command)

        # Run command.
        start_job_dqn_template_path = os.path.normpath(
            os.path.join(self.test_dir_path, "./modes/azure/grass_azure_start_job_dqn.yml"),
        )
        command = f"maro grass job start {self.cluster_name} {start_job_dqn_template_path}"
        Subprocess.run(command=command)
        self._gracefully_wait(60)

        # Check job status, failed if containers are not in running state in 120s.
        is_running = False
        start_time = time.time()
        while not is_running and start_time + 120 >= time.time():
            try:
                is_running = True
                name_to_job_details = self._get_name_to_job_details()
                self.assertTrue(len(name_to_job_details[self.test_name]["containers"]), 2)
                for _, container_details in name_to_job_details[self.test_name]["containers"].items():
                    self.assertEqual(container_details["state"]["Status"], "running")
            except AssertionError:
                is_running = False
                time.sleep(10)
        self.assertTrue(is_running)

        # Check job status, failed if containers are not in exited state in 1000s.
        is_finished = False
        start_time = time.time()
        while not is_finished and start_time + 1000 >= time.time():
            try:
                is_finished = True
                name_to_job_details = self._get_name_to_job_details()
                self.assertTrue(len(name_to_job_details[self.test_name]["containers"]), 2)
                for _, container_details in name_to_job_details[self.test_name]["containers"].items():
                    self.assertEqual(container_details["state"]["Status"], "exited")
                    self.assertEqual(container_details["state"]["ExitCode"], 0)
            except AssertionError:
                is_finished = False
                time.sleep(10)
        self.assertTrue(is_finished)

    @record_running_time(func_to_time=test_func_to_time)
    def test30_delete(self) -> None:
        # Run command.
        command = f"maro grass delete --debug {self.cluster_name}"
        Subprocess.interactive_run(command=command)


if __name__ == "__main__":
    unittest.main()
