# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
import logging
import os
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
    @classmethod
    def setUpClass(cls) -> None:
        # Get and set params
        GlobalParams.LOG_LEVEL = logging.DEBUG
        cls.test_id = uuid.uuid4().hex[:8]
        os.makedirs(os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{cls.test_id}"), exist_ok=True)
        cls.file_path = os.path.abspath(__file__)
        cls.dir_path = os.path.dirname(cls.file_path)
        cls.deployment_template_path = os.path.normpath(
            os.path.join(cls.dir_path, "../templates/test_grass_azure_create.yml"))
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

        # Pull "ubuntu" as testing image
        command = "docker pull alpine:latest"
        SubProcess.run(command)
        command = "docker pull ubuntu:latest"
        SubProcess.run(command)

    # Utils

    def _get_node_details(self) -> dict:
        command = f"maro grass node list {self.cluster_name}"
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    @staticmethod
    def _gracefully_wait(secs: int = 10) -> None:
        time.sleep(secs)

    # Tests

    def test10_create(self) -> None:
        # Run cli command
        command = f"maro grass create --debug {self.deployment_path}"
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
        # Push file to an existed folder
        command = f"maro grass data push {self.cluster_name} --debug '{self.dir_path}/test_data/README.md' '/'"
        SubProcess.interactive_run(command)

        # Push file to a new folder
        command = f"maro grass data push {self.cluster_name} --debug '{self.dir_path}/test_data/README.md' '/F1'"
        SubProcess.interactive_run(command)

        # Push folder to an existed folder
        command = f"maro grass data push {self.cluster_name} --debug '{self.dir_path}/test_data/' '/'"
        SubProcess.interactive_run(command)

        # Push folder to a new folder
        command = f"maro grass data push {self.cluster_name} --debug '{self.dir_path}/test_data/' '/F2'"
        SubProcess.interactive_run(command)

        # Pull file to an existed folder
        command = f"maro grass data pull {self.cluster_name} --debug " \
                  f"'/README.md' '{GlobalPaths.MARO_TEST}/{self.test_id}'"
        SubProcess.interactive_run(command)

        # Pull file to a new folder
        command = f"maro grass data pull {self.cluster_name} --debug " \
                  f"'/F1/README.md' '{GlobalPaths.MARO_TEST}/{self.test_id}/F1'"
        SubProcess.interactive_run(command)

        # Pull folder to an existed folder
        command = f"maro grass data pull {self.cluster_name} --debug " \
                  f"'/test_data' '{GlobalPaths.MARO_TEST}/{self.test_id}'"
        SubProcess.interactive_run(command)

        # Pull folder to a new folder
        command = f"maro grass data pull {self.cluster_name} --debug " \
                  f"'/F2/test_data/' '{GlobalPaths.MARO_TEST}/{self.test_id}/F2/'"
        SubProcess.interactive_run(command)

        self.assertTrue(os.path.exists(os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}/README.md")))
        self.assertTrue(os.path.exists(os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}/F1/README.md")))
        self.assertTrue(os.path.exists(os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}/test_data")))
        self.assertTrue(os.path.exists(os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{self.test_id}/F2/test_data")))

    def test30_delete(self) -> None:
        command = f"maro grass delete --debug {self.cluster_name}"
        SubProcess.interactive_run(command)


if __name__ == "__main__":
    unittest.main()
