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

from maro.cli.k8s.executors.k8s_aks_executor import K8sAksExecutor
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
class TestK8sCopy(unittest.TestCase):
    """Tests for K8s Copy.

    Tests should be executed in specific order,
    and the order in which the various tests will be run is determined by sorting the test method names with
    respect to the built-in ordering for strings.
    Therefore, we use test_X (X is a digit) as prefix to specify the order of the tests.
    Ref: https://docs.python.org/3.7/library/unittest.html#organizing-test-code
    """

    cluster_name = None
    test_id = None

    @classmethod
    def setUpClass(cls, file_path: str = os.path.abspath(__file__)) -> None:
        # Get and set params
        GlobalParams.LOG_LEVEL = logging.DEBUG
        cls.test_id = uuid.uuid4().hex[:8]
        os.makedirs(os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{cls.test_id}"), exist_ok=True)
        os.makedirs(os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{cls.test_id}/tar"), exist_ok=True)
        cls.file_path = os.path.abspath(__file__)
        cls.dir_path = os.path.dirname(cls.file_path)
        cls.deployment_template_path = os.path.normpath(
            os.path.join(cls.dir_path, "../templates/test_k8s_azure_create.yml"),
        )
        cls.deployment_path = os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{cls.test_id}/test_k8s_azure_create.yml")
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
        command = f"maro k8s create --debug {cls.deployment_path}"
        Subprocess.interactive_run(command=command)
        cls.cluster_details = DetailsReader.load_cluster_details(cluster_name=cls.cluster_name)
        cls.cluster_id = cls.cluster_details["id"]
        cls.executor = K8sAksExecutor(cluster_name=cls.cluster_name)
        time.sleep(15)
        cls.pod_name = cls._get_redis_pod_name()

    @classmethod
    def tearDownClass(cls) -> None:
        # Delete cluster
        command = f"maro k8s delete --debug {cls.cluster_name}"
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
    def _get_redis_pod_name(cls) -> str:
        # Get pods details
        command = "kubectl get pods -o json"
        return_str = Subprocess.run(command=command)
        pods_details = json.loads(return_str)["items"]

        # Export logs
        for pod_details in pods_details:
            if pod_details["metadata"]["labels"]["app"] == "maro-redis":
                return pod_details["metadata"]["name"]

    # Tests

    @record_speed
    def test_1_azcopy_big_file_to_remote(self) -> None:
        sas = self.executor._check_and_get_account_sas()
        command = (
            f"azcopy copy "
            f"'{self.local_big_file_path}' "
            f"'https://{self.cluster_id}st.file.core.windows.net/{self.cluster_id}-fs"
            f"/{self.test_id}/test_1_azcopy_big_file_to_remote/?{sas}' "
            f"--recursive=True"
        )
        Subprocess.interactive_run(command=command)

    @record_speed
    def test_1_azcopy_small_files_to_remote(self) -> None:
        sas = self.executor._check_and_get_account_sas()
        command = (
            f"azcopy copy "
            f"'{self.local_small_files_path}' "
            f"'https://{self.cluster_id}st.file.core.windows.net/{self.cluster_id}-fs"
            f"/{self.test_id}/test_1_azcopy_small_files_to_remote/?{sas}' "
            f"--recursive=True"
        )
        Subprocess.interactive_run(command=command)

    @record_speed
    def test_2_azcopy_big_file_to_local(self) -> None:
        sas = self.executor._check_and_get_account_sas()
        command = f"mkdir -p {GlobalPaths.MARO_TEST}/{self.test_id}/test_2_azcopy_big_file_to_local"
        _ = Subprocess.run(command=command)

        local_path = os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}/test_2_azcopy_big_file_to_local")
        command = (
            f"azcopy copy "
            f"'https://{self.cluster_id}st.file.core.windows.net/{self.cluster_id}-fs"
            f"/{self.test_id}/test_1_azcopy_big_file_to_remote?{sas}' "
            f"'{local_path}' "
            f"--recursive=True"
        )
        Subprocess.interactive_run(command=command)
        self.assertTrue(
            os.path.exists(
                os.path.expanduser(
                    f"{GlobalPaths.MARO_TEST}/{self.test_id}/"
                    f"test_2_azcopy_big_file_to_local/test_1_azcopy_big_file_to_remote/big_file",
                ),
            ),
        )

    @record_speed
    def test_2_azcopy_small_files_to_local(self) -> None:
        sas = self.executor._check_and_get_account_sas()
        command = f"mkdir -p {GlobalPaths.MARO_TEST}/{self.test_id}/test_2_azcopy_small_files_to_local"
        _ = Subprocess.run(command=command)

        local_path = os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}/test_2_azcopy_small_files_to_local")
        command = (
            f"azcopy copy "
            f"'https://{self.cluster_id}st.file.core.windows.net/{self.cluster_id}-fs"
            f"/{self.test_id}/test_1_azcopy_small_files_to_remote?{sas}' "
            f"'{local_path}' "
            f"--recursive=True"
        )
        Subprocess.interactive_run(command=command)
        self.assertTrue(
            os.path.exists(
                os.path.expanduser(
                    f"{GlobalPaths.MARO_TEST}/{self.test_id}/"
                    f"test_2_azcopy_small_files_to_local/test_1_azcopy_small_files_to_remote/small_files",
                ),
            ),
        )

    @record_speed
    def test_1_kubectl_exec_big_file_to_remote(self) -> None:
        command = (
            f"kubectl exec -i {self.pod_name} -- "
            f"mkdir -p /mnt/maro/{self.test_id}/test_1_kubectl_exec_big_file_to_remote"
        )
        Subprocess.interactive_run(command=command)

        basename = os.path.basename(self.local_big_file_path)
        dirname = os.path.dirname(self.local_big_file_path)
        command = (
            f"tar cf - -C {dirname} {basename} | "
            f"kubectl exec -i {self.pod_name} -- "
            f"tar xf - -C /mnt/maro/{self.test_id}/test_1_kubectl_exec_big_file_to_remote"
        )
        Subprocess.interactive_run(command=command)

    @record_speed
    def test_1_kubectl_exec_small_files_to_remote(self) -> None:
        command = (
            f"kubectl exec -i {self.pod_name} -- "
            f"mkdir -p /mnt/maro/{self.test_id}/test_1_kubectl_exec_small_files_to_remote"
        )
        Subprocess.interactive_run(command=command)

        basename = os.path.basename(self.local_small_files_path)
        dirname = os.path.dirname(self.local_small_files_path)
        command = (
            f"tar cf - -C {dirname} {basename} | "
            f"kubectl exec -i {self.pod_name} -- "
            f"tar xf - -C /mnt/maro/{self.test_id}/test_1_kubectl_exec_small_files_to_remote"
        )
        Subprocess.interactive_run(command=command)

    @record_speed
    def test_2_kubectl_exec_big_file_to_local(self) -> None:
        command = f"mkdir -p {GlobalPaths.MARO_TEST}/{self.test_id}/test_2_kubectl_exec_big_file_to_local"
        _ = Subprocess.run(command=command)

        basename = os.path.basename(f"/mnt/maro/{self.test_id}/test_1_kubectl_exec_big_file_to_remote")
        dirname = os.path.dirname(f"/mnt/maro/{self.test_id}/test_1_kubectl_exec_big_file_to_remote")
        command = (
            f"kubectl exec -i {self.pod_name} -- tar cf - -C {dirname} {basename}  | "
            f"tar xf - -C {GlobalPaths.MARO_TEST}/{self.test_id}/test_2_kubectl_exec_big_file_to_local"
        )
        Subprocess.interactive_run(command=command)
        self.assertTrue(
            os.path.exists(
                os.path.expanduser(
                    f"{GlobalPaths.MARO_TEST}/{self.test_id}/"
                    f"test_2_kubectl_exec_big_file_to_local/test_1_kubectl_exec_big_file_to_remote/big_file",
                ),
            ),
        )

    @record_speed
    def test_2_kubectl_exec_small_files_to_local(self) -> None:
        command = f"mkdir -p {GlobalPaths.MARO_TEST}/{self.test_id}/test_2_kubectl_exec_small_files_to_local"
        _ = Subprocess.run(command=command)

        basename = os.path.basename(f"/mnt/maro/{self.test_id}/test_1_kubectl_exec_small_files_to_remote")
        dirname = os.path.dirname(f"/mnt/maro/{self.test_id}/test_1_kubectl_exec_small_files_to_remote")
        command = (
            f"kubectl exec -i {self.pod_name} -- tar cf - -C {dirname} {basename}  | "
            f"tar xf - -C {GlobalPaths.MARO_TEST}/{self.test_id}/test_2_kubectl_exec_small_files_to_local"
        )
        Subprocess.interactive_run(command=command)
        self.assertTrue(
            os.path.exists(
                os.path.expanduser(
                    f"{GlobalPaths.MARO_TEST}/{self.test_id}/"
                    f"test_2_kubectl_exec_small_files_to_local/test_1_kubectl_exec_small_files_to_remote/small_files",
                ),
            ),
        )

    @record_speed
    def test_1_azcopy_tar_big_file_to_remote(self) -> None:
        # create remote folder
        command = (
            f"kubectl exec -i {self.pod_name} -- "
            f"mkdir -p /mnt/maro/{self.test_id}/test_1_azcopy_tar_big_file_to_remote"
        )
        Subprocess.interactive_run(command=command)

        # local tar zip
        basename = os.path.basename(self.local_big_file_path)
        dirname = os.path.dirname(self.local_big_file_path)
        tar_file_name = uuid.uuid4().hex[:8]
        command = f"tar cf {GlobalPaths.MARO_TEST}/{self.test_id}/tar/{tar_file_name} -C {dirname} {basename}"
        Subprocess.interactive_run(command=command)

        # azcopy
        sas = self.executor._check_and_get_account_sas()
        local_path = os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}/tar/{tar_file_name}")
        command = (
            f"azcopy copy "
            f"'{local_path}' "
            f"'https://{self.cluster_id}st.file.core.windows.net/{self.cluster_id}-fs"
            f"/tar/{tar_file_name}?{sas}' "
            f"--recursive=True"
        )
        Subprocess.interactive_run(command=command)

        # remote tar unzip
        command = (
            f"kubectl exec -i {self.pod_name} -- "
            f"tar xf /mnt/maro/tar/{tar_file_name} "
            f"-C /mnt/maro/{self.test_id}/test_1_azcopy_tar_big_file_to_remote"
        )
        Subprocess.interactive_run(command=command)

    @record_speed
    def test_1_azcopy_tar_small_files_to_remote(self) -> None:
        # create remote folder
        command = (
            f"kubectl exec -i {self.pod_name} -- "
            f"mkdir -p /mnt/maro/{self.test_id}/test_1_azcopy_tar_small_files_to_remote"
        )
        Subprocess.interactive_run(command=command)

        # local tar zip
        basename = os.path.basename(self.local_small_files_path)
        dirname = os.path.dirname(self.local_small_files_path)
        tar_file_name = uuid.uuid4().hex[:8]
        command = f"tar cf {GlobalPaths.MARO_TEST}/{self.test_id}/tar/{tar_file_name} -C {dirname} {basename}"
        Subprocess.interactive_run(command=command)

        # azcopy
        sas = self.executor._check_and_get_account_sas()
        local_path = os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}/tar/{tar_file_name}")
        command = (
            f"azcopy copy "
            f"'{local_path}' "
            f"'https://{self.cluster_id}st.file.core.windows.net/{self.cluster_id}-fs"
            f"/tar/{tar_file_name}?{sas}' "
            f"--recursive=True"
        )
        Subprocess.interactive_run(command=command)

        # remote tar unzip
        command = (
            f"kubectl exec -i {self.pod_name} -- "
            f"tar xf /mnt/maro/tar/{tar_file_name} "
            f"-C /mnt/maro/{self.test_id}/test_1_azcopy_tar_small_files_to_remote"
        )
        Subprocess.interactive_run(command=command)

    @record_speed
    def test_2_azcopy_tar_big_file_to_local(self) -> None:
        # create folder
        command = f"mkdir -p {GlobalPaths.MARO_TEST}/{self.test_id}/test_2_azcopy_tar_big_file_to_local"
        Subprocess.interactive_run(command=command)

        # remote tar zip
        basename = os.path.basename(f"/mnt/maro/{self.test_id}/test_1_azcopy_tar_big_file_to_remote")
        dirname = os.path.dirname(f"/mnt/maro/{self.test_id}/test_1_azcopy_tar_big_file_to_remote")
        tar_file_name = uuid.uuid4().hex[:8]
        command = f"kubectl exec -i {self.pod_name} -- " f"tar cf /mnt/maro/tar/{tar_file_name} -C {dirname} {basename}"
        Subprocess.interactive_run(command=command)

        # azcopy
        sas = self.executor._check_and_get_account_sas()
        local_path = os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}/tar/{tar_file_name}")
        command = (
            f"azcopy copy "
            f"'https://{self.cluster_id}st.file.core.windows.net/{self.cluster_id}-fs"
            f"/tar/{tar_file_name}?{sas}' "
            f"'{local_path}' "
            f"--recursive=True"
        )
        Subprocess.interactive_run(command=command)

        # local tar unzip
        command = (
            f"tar xf {GlobalPaths.MARO_TEST}/{self.test_id}/tar/{tar_file_name} "
            f"-C {GlobalPaths.MARO_TEST}/{self.test_id}/test_2_azcopy_tar_big_file_to_local"
        )
        Subprocess.interactive_run(command=command)
        self.assertTrue(
            os.path.exists(
                os.path.expanduser(
                    f"{GlobalPaths.MARO_TEST}/{self.test_id}/"
                    f"test_2_azcopy_tar_big_file_to_local/test_1_azcopy_tar_big_file_to_remote/big_file",
                ),
            ),
        )

    @record_speed
    def test_2_azcopy_tar_small_files_to_local(self) -> None:
        # create folder
        command = f"mkdir -p {GlobalPaths.MARO_TEST}/{self.test_id}/test_2_azcopy_tar_small_files_to_local"
        Subprocess.interactive_run(command=command)

        # remote tar zip
        basename = os.path.basename(f"/mnt/maro/{self.test_id}/test_1_azcopy_tar_small_files_to_remote")
        dirname = os.path.dirname(f"/mnt/maro/{self.test_id}/test_1_azcopy_tar_small_files_to_remote")
        tar_file_name = uuid.uuid4().hex[:8]
        command = f"kubectl exec -i {self.pod_name} -- " f"tar cf /mnt/maro/tar/{tar_file_name} -C {dirname} {basename}"
        Subprocess.interactive_run(command=command)

        # azcopy
        sas = self.executor._check_and_get_account_sas()
        local_path = os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}/tar/{tar_file_name}")
        command = (
            f"azcopy copy "
            f"'https://{self.cluster_id}st.file.core.windows.net/{self.cluster_id}-fs/tar/{tar_file_name}?{sas}' "
            f"'{local_path}' "
            f"--recursive=True"
        )
        Subprocess.interactive_run(command=command)

        # local tar unzip
        command = (
            f"tar xf {GlobalPaths.MARO_TEST}/{self.test_id}/tar/{tar_file_name} "
            f"-C {GlobalPaths.MARO_TEST}/{self.test_id}/test_2_azcopy_tar_small_files_to_local"
        )
        Subprocess.interactive_run(command=command)
        self.assertTrue(
            os.path.exists(
                os.path.expanduser(
                    f"{GlobalPaths.MARO_TEST}/{self.test_id}/"
                    f"test_2_azcopy_tar_small_files_to_local/test_1_azcopy_tar_small_files_to_remote/small_files",
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
