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

from maro.cli.utils.params import GlobalParams, GlobalPaths
from maro.cli.utils.subprocess import SubProcess


@unittest.skipUnless(os.environ.get("test_with_cli", False), "Require cli prerequisites.")
class TestGrass(unittest.TestCase):
    """Tests for Grass Mode.

    Tests should be executed in specific order,
    and the order in which the various tests will be run is determined by sorting the test method names with
    respect to the built-in ordering for strings.
    We use testXX (X is a digit) as prefix to specify the order of the tests.
    Ref: https://docs.python.org/3.7/library/unittest.html#organizing-test-code
    """
    test_id = None

    @classmethod
    def setUpClass(cls) -> None:
        # Get and set params
        GlobalParams.LOG_LEVEL = logging.DEBUG
        cls.test_id = uuid.uuid4().hex[:8]
        os.makedirs(os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{cls.test_id}"), exist_ok=True)
        cls.test_file_path = os.path.abspath(__file__)
        cls.test_dir_path = os.path.dirname(cls.test_file_path)
        cls.maro_pkg_path = os.path.normpath(os.path.join(cls.test_file_path, "../../../../"))
        cls.create_deployment_template_path = os.path.normpath(
            os.path.join(cls.test_dir_path, "../templates/test_grass_azure_create.yml"))
        cls.create_deployment_path = os.path.expanduser(
            f"{GlobalPaths.MARO_TEST}/{cls.test_id}/test_grass_azure_create.yml")
        cls.test_config_path = os.path.normpath(os.path.join(cls.test_dir_path, "../config.yml"))

        # Load config and save deployment
        with open(cls.create_deployment_template_path) as fr:
            create_deployment_details = yaml.safe_load(fr)
        with open(cls.test_config_path) as fr:
            test_config_details = yaml.safe_load(fr)
            if test_config_details["cloud/subscription"] and test_config_details["user/admin_public_key"]:
                create_deployment_details["cloud"]["subscription"] = test_config_details["cloud/subscription"]
                create_deployment_details["user"]["admin_public_key"] = test_config_details["user/admin_public_key"]
            else:
                raise Exception("Invalid config")
        with open(cls.create_deployment_path, "w") as fw:
            yaml.safe_dump(create_deployment_details, fw)

        # Get params from deployments
        cls.cluster_name = create_deployment_details["name"]

        # Pull "ubuntu" as testing image
        command = "docker pull alpine:latest"
        SubProcess.run(command)
        command = "docker pull ubuntu:latest"
        SubProcess.run(command)

    @classmethod
    def tearDownClass(cls) -> None:
        # Delete tmp test folder
        shutil.rmtree(os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{cls.test_id}"))

    # Utils

    def _get_node_details(self) -> dict:
        command = f"maro grass node list {self.cluster_name}"
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    def _get_master_details(self) -> dict:
        command = f"maro grass status {self.cluster_name} master"
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    def _list_jobs_details(self) -> dict:
        command = f"maro grass job list {self.cluster_name}"
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    @staticmethod
    def _gracefully_wait(secs: int = 10) -> None:
        time.sleep(secs)

    # Tests

    def test10_create(self) -> None:
        # Run cli command
        command = f"maro grass create --debug {self.create_deployment_path}"
        SubProcess.interactive_run(command)

    def test11_node1(self) -> None:
        """Scale node spec Standard_D4s_v3 to 1.

        A Standard_D4s_v3 should be in running state.

        Returns:
            None.
        """
        # Run cli command
        command = f"maro grass node scale {self.cluster_name} --debug Standard_D4s_v3 1"
        SubProcess.interactive_run(command)

        # Check validity
        nodes_details = self._get_node_details()
        self.assertEqual(len(nodes_details), 1)
        for _, node_details in nodes_details.items():
            self.assertEqual("Running", node_details["state"])

    def test12_image1(self) -> None:
        """Push image alpine:latest to the cluster.

        The only Standard_D4s_v3 should have loaded the image alpine:latest.

        Returns:
            None.
        """
        # Run cli command
        command = f"maro grass image push {self.cluster_name} --debug --image-name alpine:latest"
        SubProcess.interactive_run(command)
        self._gracefully_wait()

        # Check validity
        nodes_details = self._get_node_details()
        self.assertEqual(len(nodes_details), 1)
        for _, node_details in nodes_details.items():
            self.assertEqual("Running", node_details["state"])
            self.assertIn("alpine_latest", node_details["image_files"])

    def test13_node2(self) -> None:
        """Scale node spec Standard_D4s_v3 to 2.

        Two Standard_D4s_v3 should be in running state, and they should have loaded the image alpine:latest.

        Returns:
            None.
        """
        # Run cli command
        command = f"maro grass node scale {self.cluster_name} --debug Standard_D4s_v3 2"
        SubProcess.interactive_run(command)
        self._gracefully_wait()

        # Check validity
        nodes_details = self._get_node_details()
        self.assertEqual(len(nodes_details), 2)
        for _, node_details in nodes_details.items():
            self.assertEqual("Running", node_details["state"])
            self.assertIn("alpine_latest", node_details["image_files"])

    def test14_stop(self) -> None:
        """Stop one Standard_D4s_v3.

        One Standard_D4s_v3 should be in running state, and the other should be in Stopped state.

        Returns:
            None.
        """
        # Run cli command
        command = f"maro grass node stop {self.cluster_name} --debug Standard_D4s_v3 1"
        SubProcess.interactive_run(command)
        self._gracefully_wait()

        # Check validity
        nodes_details = self._get_node_details()
        self.assertEqual(len(nodes_details), 2)
        running_count = 0
        stopped_count = 0
        for _, node_details in nodes_details.items():
            if node_details["state"] == "Running":
                running_count += 1
            if node_details["state"] == "Stopped":
                stopped_count += 1
        self.assertEqual(running_count, 1)
        self.assertEqual(stopped_count, 1)

    def test15_image2(self) -> None:
        """Push image ubuntu:latest to the cluster.

        The only Running Standard_D4s_v3 should have loaded the image ubuntu:latest,
        and the other should have not.

        Returns:
            None.
        """
        # Run cli command
        command = f"maro grass image push {self.cluster_name} --debug --image-name ubuntu:latest"
        SubProcess.interactive_run(command)
        self._gracefully_wait()

        # Check validity
        nodes_details = self._get_node_details()
        self.assertEqual(len(nodes_details), 2)
        running_count = 0
        stopped_count = 0
        for _, node_details in nodes_details.items():
            if node_details["state"] == "Running":
                running_count += 1
                self.assertIn("alpine_latest", node_details["image_files"])
                self.assertIn("ubuntu_latest", node_details["image_files"])
            if node_details["state"] == "Stopped":
                stopped_count += 1
                self.assertIn("alpine_latest", node_details["image_files"])
                self.assertNotIn("ubuntu_latest", node_details["image_files"])
        self.assertEqual(running_count, 1)
        self.assertEqual(stopped_count, 1)

    def test16_start(self) -> None:
        """Start one Standard_D4s_v3.

        Two Standard_D4s_v3 should be in running state,
        and they should have loaded the image alpine:latest and ubuntu:latest.

        Returns:
            None.
        """
        command = f"maro grass node start {self.cluster_name} --debug Standard_D4s_v3 1"
        SubProcess.interactive_run(command)
        self._gracefully_wait()

        # Check validity
        nodes_details = self._get_node_details()
        self.assertEqual(len(nodes_details), 2)
        running_count = 0
        for _, node_details in nodes_details.items():
            if node_details["state"] == "Running":
                running_count += 1
                self.assertIn("alpine_latest", node_details["image_files"])
                self.assertIn("ubuntu_latest", node_details["image_files"])
        self.assertEqual(running_count, 2)

    def test17_data(self) -> None:
        # Create tmp files
        test_dir = os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}")
        os.makedirs(f"{test_dir}/push/test_data", exist_ok=True)
        os.makedirs(f"{test_dir}/pull", exist_ok=True)
        command = f"dd if=/dev/zero of={test_dir}/push/test_data/a.file bs=1 count=0 seek=1M"
        SubProcess.run(command)

        # Push file to an existed folder
        command = (f"maro grass data push {self.cluster_name} --debug "
                   f"'{GlobalPaths.MARO_TEST}/{self.test_id}/push/test_data/a.file' '/'")
        SubProcess.interactive_run(command)

        # Push file to a new folder
        command = (f"maro grass data push {self.cluster_name} --debug "
                   f"'{GlobalPaths.MARO_TEST}/{self.test_id}/push/test_data/a.file' '/F1'")
        SubProcess.interactive_run(command)

        # Push folder to an existed folder
        command = (f"maro grass data push {self.cluster_name} --debug "
                   f"'{GlobalPaths.MARO_TEST}/{self.test_id}/push/test_data/' '/'")
        SubProcess.interactive_run(command)

        # Push folder to a new folder
        command = (f"maro grass data push {self.cluster_name} --debug "
                   f"'{GlobalPaths.MARO_TEST}/{self.test_id}/push/test_data/' '/F2'")
        SubProcess.interactive_run(command)

        # Pull file to an existed folder
        command = (f"maro grass data pull {self.cluster_name} --debug "
                   f"'/a.file' '{GlobalPaths.MARO_TEST}/{self.test_id}/pull'")
        SubProcess.interactive_run(command)

        # Pull file to a new folder
        command = (f"maro grass data pull {self.cluster_name} --debug "
                   f"'/F1/a.file' '{GlobalPaths.MARO_TEST}/{self.test_id}/pull/F1'")
        SubProcess.interactive_run(command)

        # Pull folder to an existed folder
        command = (f"maro grass data pull {self.cluster_name} --debug "
                   f"'/test_data' '{GlobalPaths.MARO_TEST}/{self.test_id}/pull'")
        SubProcess.interactive_run(command)

        # Pull folder to a new folder
        command = (f"maro grass data pull {self.cluster_name} --debug "
                   f"'/F2/test_data/' '{GlobalPaths.MARO_TEST}/{self.test_id}/pull/F2/'")
        SubProcess.interactive_run(command)

        self.assertTrue(os.path.exists(os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}/pull/a.file")))
        self.assertTrue(os.path.exists(os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}/pull/F1/a.file")))
        self.assertTrue(os.path.exists(os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}/pull/test_data")))
        self.assertTrue(os.path.exists(os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}/pull/F2/test_data")))

    def test20_train_dqn(self) -> None:
        # Copy dqn examples to test folder
        dqn_source_dir = os.path.normpath(os.path.join(self.test_dir_path, "../../../examples/cim/dqn"))
        dqn_target_dir = os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}/train/dqn")
        os.makedirs(os.path.dirname(dqn_target_dir), exist_ok=True)
        command = f"cp -r {dqn_source_dir} {dqn_target_dir}"
        SubProcess.run(command)

        # Get cluster details and rebuild config
        master_details = self._get_master_details()
        with open(f"{dqn_target_dir}/config.yml", 'r') as fr:
            config = yaml.safe_load(fr)
        with open(f"{dqn_target_dir}/config.yml", 'w') as fw:
            config["general"]["total_training_episodes"] = 30
            config["distributed"]["group_name"] = self.test_id
            config["distributed"]["redis"]["host_name"] = master_details["private_ip_address"]
            config["distributed"]["redis"]["port"] = master_details["redis"]["port"]
            yaml.safe_dump(config, fw)

        # Push dqn folder to cluster
        command = (f"maro grass data push {self.cluster_name} --debug "
                   f"'{GlobalPaths.MARO_TEST}/{self.test_id}/train/dqn' '/train'")
        SubProcess.run(command)

        # Build docker image and load docker image
        command = (f"docker build -f {self.maro_pkg_path}/docker_files/cpu.runtime.df -t maro_runtime_cpu:test "
                   f"{self.maro_pkg_path}")
        SubProcess.run(command)
        command = f"maro grass image push {self.cluster_name} --debug --image-name maro_runtime_cpu:test"
        SubProcess.interactive_run(command)

        # Start job
        start_job_dqn_template_path = os.path.normpath(
            os.path.join(self.test_dir_path, "../templates/test_grass_azure_start_job_dqn.yml"))
        command = f"maro grass job start {self.cluster_name} {start_job_dqn_template_path}"
        SubProcess.run(command)
        self._gracefully_wait(30)

        # Check job status
        remain_idx = 0
        is_finished = True
        while remain_idx <= 100:
            is_finished = True
            jobs_details = self._list_jobs_details()
            self.assertTrue(len(jobs_details["job_for_test"]["containers"]), 2)
            for _, container_details in jobs_details["job_for_test"]["containers"].items():
                if container_details["state"]["Status"] == "running":
                    is_finished = False
            if is_finished:
                break
            time.sleep(10)
            remain_idx += 1
        self.assertTrue(is_finished)

    def test30_delete(self) -> None:
        command = f"maro grass delete --debug {self.cluster_name}"
        SubProcess.interactive_run(command)


if __name__ == "__main__":
    unittest.main()
