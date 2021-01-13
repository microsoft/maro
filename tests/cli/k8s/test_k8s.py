# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import ast
import json
import logging
import os
import platform
import shutil
import time
import unittest
import uuid

import yaml

from maro.cli.utils.azure_controller import AzureController
from maro.cli.utils.params import GlobalParams, GlobalPaths
from maro.cli.utils.subprocess import Subprocess
from tests.cli.utils import record_running_time


@unittest.skipUnless(os.environ.get("test_with_cli", False), "Require cli prerequisites.")
class TestK8s(unittest.TestCase):
    """Tests for K8s Mode.

    Tests should be executed in specific order,
    and the order in which the various tests will be run is determined by sorting the test method names with
    respect to the built-in ordering for strings.
    We use testXX (X is a digit) as prefix to specify the order of the tests.
    Ref: https://docs.python.org/3.7/library/unittest.html#organizing-test-code
    """
    test_id = None
    job_name = "job1"
    test_func_to_time = {}
    cluster_name = None
    resource_group = None

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
            os.path.join(cls.test_dir_path, "../templates/test_k8s_azure_create.yml")
        )
        cls.create_deployment_path = os.path.expanduser(
            f"{GlobalPaths.MARO_TEST}/{cls.test_id}/test_k8s_azure_create.yml"
        )
        cls.test_config_path = os.path.normpath(os.path.join(cls.test_dir_path, "../config.yml"))

        # Load config and save deployment
        with open(cls.create_deployment_template_path) as fr:
            create_deployment_details = yaml.safe_load(fr)
        with open(cls.test_config_path) as fr:
            config_details = yaml.safe_load(fr)
            if config_details["cloud/subscription"] and config_details["user/admin_public_key"]:
                create_deployment_details["name"] = f"test_maro_k8s_{cls.test_id}"
                create_deployment_details["cloud"]["subscription"] = config_details["cloud/subscription"]
                create_deployment_details["cloud"]["resource_group"] = f"test_maro_k8s_{cls.test_id}"
                create_deployment_details["user"]["admin_public_key"] = config_details["user/admin_public_key"]
            else:
                raise Exception("Invalid config")
        with open(cls.create_deployment_path, "w") as fw:
            yaml.safe_dump(create_deployment_details, fw)

        # Get params from deployments
        cls.cluster_name = create_deployment_details["name"]
        cls.resource_group = create_deployment_details["cloud"]["resource_group"]

        # Pull "ubuntu" as testing image
        command = "docker pull alpine:latest"
        Subprocess.run(command=command)
        command = "docker pull ubuntu:latest"
        Subprocess.run(command=command)

    @classmethod
    def tearDownClass(cls) -> None:
        # Print result
        print(
            json.dumps(
                cls.test_func_to_time,
                indent=4, sort_keys=True
            )
        )

        # Delete resource group
        AzureController.delete_resource_group(resource_group=cls.resource_group)

        # Delete tmp test folder
        shutil.rmtree(os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{cls.test_id}"))

    # Utils

    def _get_node_details(self) -> dict:
        command = f"maro k8s node list {self.cluster_name}"
        return_str = Subprocess.run(command=command)
        return json.loads(return_str)

    def _get_image_details(self) -> dict:
        command = f"maro k8s image list {self.cluster_name}"
        return_str = Subprocess.run(command=command)
        return json.loads(return_str)

    def _get_cluster_details(self) -> dict:
        command = f"maro k8s status {self.cluster_name}"
        return_str = Subprocess.run(command=command)
        return json.loads(return_str)

    def _list_jobs(self) -> dict:
        command = f"maro k8s job list {self.cluster_name}"
        return_str = Subprocess.run(command=command)
        return json.loads(return_str)

    def _get_name_to_job_details(self) -> dict:
        jobs = self._list_jobs()
        name_to_job_details = {}
        for job in jobs:
            name_to_job_details[job["metadata"]["labels"]["jobName"]] = job
        return name_to_job_details

    @staticmethod
    def _gracefully_wait(secs: int = 10) -> None:
        time.sleep(secs)

    # Tests

    @record_running_time(func_to_time=test_func_to_time)
    def test10_create(self) -> None:
        # Run cli command
        command = f"maro k8s create --debug {self.create_deployment_path}"
        Subprocess.interactive_run(command=command)

        # Check validity
        nodes_details = self._get_node_details()
        self.assertIn("Standard_D2s_v3", nodes_details)
        self.assertEqual(nodes_details["Standard_D2s_v3"], 1)

    @record_running_time(func_to_time=test_func_to_time)
    def test11_node(self) -> None:
        # Run cli command
        command = f"maro k8s node scale {self.cluster_name} --debug Standard_D4s_v3 1"
        Subprocess.interactive_run(command=command)

        # Check validity
        nodes_details = self._get_node_details()
        self.assertIn("Standard_D2s_v3", nodes_details)
        self.assertIn("Standard_D4s_v3", nodes_details)
        self.assertEqual(nodes_details["Standard_D2s_v3"], 1)
        self.assertEqual(nodes_details["Standard_D4s_v3"], 1)

    @record_running_time(func_to_time=test_func_to_time)
    def test12_image(self) -> None:
        # Run cli command
        command = f"maro k8s image push {self.cluster_name} --debug --image-name alpine:latest"
        Subprocess.interactive_run(command=command)

        # Check validity
        command = f"maro k8s image list {self.cluster_name}"
        return_str = Subprocess.run(command=command)
        images = ast.literal_eval(return_str)
        self.assertIn("alpine", images)

    @record_running_time(func_to_time=test_func_to_time)
    def test13_data(self) -> None:
        # Create tmp files
        test_dir = os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}")
        os.makedirs(f"{test_dir}/push/test_data", exist_ok=True)
        os.makedirs(f"{test_dir}/pull", exist_ok=True)
        if platform.system() == "Windows":
            command = f"fsutil file createnew {test_dir}/push/test_data/a.file 1048576"
        else:
            command = f"fallocate -l 1M {test_dir}/push/test_data/a.file"
        Subprocess.run(command=command)

        # Push file to an existed folder
        command = (
            f"maro k8s data push {self.cluster_name} --debug "
            f"'{GlobalPaths.MARO_TEST}/{self.test_id}/push/test_data/a.file' '/'"
        )
        Subprocess.interactive_run(command=command)

        # Push file to a new folder
        command = (
            f"maro k8s data push {self.cluster_name} --debug "
            f"'{GlobalPaths.MARO_TEST}/{self.test_id}/push/test_data/a.file' '/F1'"
        )
        Subprocess.interactive_run(command=command)

        # Push folder to an existed folder
        command = (
            f"maro k8s data push {self.cluster_name} --debug "
            f"'{GlobalPaths.MARO_TEST}/{self.test_id}/push/test_data/' '/'"
        )
        Subprocess.interactive_run(command=command)

        # Push folder to a new folder
        command = (
            f"maro k8s data push {self.cluster_name} --debug "
            f"'{GlobalPaths.MARO_TEST}/{self.test_id}/push/test_data/' '/F2'"
        )
        Subprocess.interactive_run(command=command)

        # Pull file to an existed folder
        command = (
            f"maro k8s data pull {self.cluster_name} --debug "
            f"'/a.file' '{GlobalPaths.MARO_TEST}/{self.test_id}/pull'"
        )
        Subprocess.interactive_run(command=command)

        # Pull file to a new folder
        command = (
            f"maro k8s data pull {self.cluster_name} --debug "
            f"'/F1/a.file' '{GlobalPaths.MARO_TEST}/{self.test_id}/pull/F1'"
        )
        Subprocess.interactive_run(command=command)

        # Pull folder to an existed folder
        command = (
            f"maro k8s data pull {self.cluster_name} --debug "
            f"'/test_data' '{GlobalPaths.MARO_TEST}/{self.test_id}/pull'"
        )
        Subprocess.interactive_run(command=command)

        # Pull folder to a new folder
        command = (
            f"maro k8s data pull {self.cluster_name} --debug "
            f"'/F2/test_data/' '{GlobalPaths.MARO_TEST}/{self.test_id}/pull/F2/'"
        )
        Subprocess.interactive_run(command=command)

        self.assertTrue(os.path.exists(os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}/pull/a.file")))
        self.assertTrue(os.path.exists(os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}/pull/F1/a.file")))
        self.assertTrue(os.path.exists(os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}/pull/test_data")))
        self.assertTrue(os.path.exists(os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}/pull/F2/test_data")))

    @record_running_time(func_to_time=test_func_to_time)
    def test20_train_env_provision(self):
        # Build docker image and load docker image
        command = (
            f"docker build -f {self.maro_pkg_path}/docker_files/cpu.runtime.source.df -t maro_runtime_cpu "
            f"{self.maro_pkg_path}"
        )
        Subprocess.run(command=command)
        command = f"maro k8s image push {self.cluster_name} --debug --image-name maro_runtime_cpu"
        Subprocess.interactive_run(command=command)

    @record_running_time(func_to_time=test_func_to_time)
    def test21_train_dqn(self) -> None:
        # Copy dqn examples to test folder
        dqn_source_dir = os.path.normpath(os.path.join(self.test_dir_path, "../../../examples/cim/dqn"))
        dqn_target_dir = os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}/train/dqn")
        os.makedirs(os.path.dirname(dqn_target_dir), exist_ok=True)
        command = f"cp -r {dqn_source_dir} {dqn_target_dir}"
        Subprocess.run(command=command)

        # Get cluster details and rebuild config
        cluster_details = self._get_cluster_details()
        with open(f"{dqn_target_dir}/config.yml", 'r') as fr:
            config = yaml.safe_load(fr)
        with open(f"{dqn_target_dir}/distributed_config.yml", "r") as fr:
            distributed_config = yaml.safe_load(fr)
        with open(f"{dqn_target_dir}/config.yml", "w") as fw:
            config["general"]["max_episode"] = 30
            yaml.safe_dump(config, fw)
        with open(f"{dqn_target_dir}/distributed_config.yml", 'w') as fw:
            distributed_config["redis"]["hostname"] = cluster_details["redis"]["private_ip_address"]
            yaml.safe_dump(distributed_config, fw)

        # Push dqn folder to cluster
        command = (
            f"maro k8s data push {self.cluster_name} --debug "
            f"'{GlobalPaths.MARO_TEST}/{self.test_id}/train/dqn' '/train'"
        )
        Subprocess.run(command=command)

        # Start job
        start_job_dqn_template_path = os.path.normpath(
            os.path.join(self.test_dir_path, "../templates/test_k8s_azure_start_job_dqn.yml")
        )
        command = f"maro k8s job start {self.cluster_name} {start_job_dqn_template_path}"
        Subprocess.run(command=command)
        self._gracefully_wait(60)

        # Check job status
        remain_idx = 0
        is_finished = False
        while remain_idx <= 100:
            name_to_job_details = self._get_name_to_job_details()
            job_details = name_to_job_details[self.job_name]
            if "succeeded" in job_details["status"] and job_details["status"]["succeeded"] == 1:
                is_finished = True
                break
            time.sleep(10)
            remain_idx += 1
        self.assertTrue(is_finished)

    @record_running_time(func_to_time=test_func_to_time)
    def test30_delete(self) -> None:
        command = f"maro k8s delete --debug {self.cluster_name}"
        Subprocess.interactive_run(command=command)


if __name__ == "__main__":
    unittest.main()
