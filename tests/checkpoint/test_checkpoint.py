# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
import time
import unittest
import uuid

import yaml

from maro.cli.utils.azure_controller import AzureController
from maro.cli.utils.params import GlobalParams, GlobalPaths
from maro.utils.checkpoint import AzureBlobCheckpoint, ServerCheckpoint


@unittest.skipUnless(os.environ.get("test_with_checkpoint", False), "Require checkpoint prerequisites.")
class TestCheckPoint(unittest.TestCase):
    resource_group = "test_maro_checkpoint"
    location = "japaneast"
    admin_username = "marotest"

    @classmethod
    def setUpClass(cls) -> None:
        # Get and set params
        GlobalParams.LOG_LEVEL = logging.DEBUG
        cls.test_id = uuid.uuid4().hex[:8]
        os.makedirs(os.path.expanduser(f"{GlobalPaths.MARO_TEST}/{cls.test_id}"), exist_ok=True)
        cls.test_file_path = os.path.abspath(__file__)
        cls.test_dir_path = os.path.dirname(cls.test_file_path)

        # Load config
        cls.config_path = os.path.normpath(os.path.join(cls.test_dir_path, "./config.yml"))

        # Load config
        with open(cls.config_path) as fr:
            config_details = yaml.safe_load(fr)
            if config_details["cloud/subscription"] and config_details["user/admin_public_key"]:
                pass
            else:
                raise Exception("Invalid config")

        # Create resource group
        AzureController.create_resource_group(cls.resource_group, cls.location)

        # Create ARM params
        template_file_location = f"{cls.test_dir_path}/test_checkpoint_template.json"
        base_parameters_file_location = f"{cls.test_dir_path}/test_checkpoint_parameters.json"
        parameters_file_location = os.path.expanduser(
            f"{GlobalPaths.MARO_TEST}/{cls.test_id}/test_checkpoint_parameters.json",
        )
        with open(base_parameters_file_location, "r") as f:
            base_parameters = json.load(f)
        with open(parameters_file_location, "w") as fw:
            parameters = base_parameters["parameters"]
            parameters["location"]["value"] = cls.location
            parameters["networkInterfaceName"]["value"] = f"{cls.test_id}-nic"
            parameters["networkSecurityGroupName"]["value"] = f"{cls.test_id}-nsg"
            parameters["virtualNetworkName"]["value"] = f"{cls.test_id}-vnet"
            parameters["publicIpAddressName"]["value"] = f"{cls.test_id}-pip"
            parameters["virtualMachineName"]["value"] = f"{cls.test_id}-vm"
            parameters["virtualMachineSize"]["value"] = "Standard_B2s"
            parameters["adminUsername"]["value"] = cls.admin_username
            parameters["adminPublicKey"]["value"] = config_details["user/admin_public_key"]
            parameters["storageAccountName"]["value"] = f"{cls.test_id}st"
            json.dump(base_parameters, fw, indent=4)

        # Start ARM deployment
        AzureController.start_deployment(
            resource_group=cls.resource_group,
            deployment_name=cls.test_id,
            template_file=template_file_location,
            parameters_file=parameters_file_location,
        )
        cls._gracefully_wait(15)

        # Get params after ARM deployment
        cls.conn_str = AzureController.get_connection_string(storage_account_name=f"{cls.test_id}st")
        ip_addresses = AzureController.list_ip_addresses(resource_group=cls.resource_group, vm_name=f"{cls.test_id}-vm")
        cls.ip_address = ip_addresses[0]["virtualMachine"]["network"]["publicIpAddresses"][0]["ipAddress"]

    @classmethod
    def tearDownClass(cls) -> None:
        # Delete resource group after the test
        AzureController.delete_resource_group(cls.resource_group)

    # Utils

    @staticmethod
    def _gracefully_wait(secs: int = 10) -> None:
        time.sleep(secs)

    # tests

    def test_azure_blob_checkpoint(self):
        checkpoint = AzureBlobCheckpoint(conn_str=self.conn_str, container_name="test-container")
        self.assertFalse(checkpoint.exists("key1"))
        checkpoint.set("key1", b"a1234")
        checkpoint.set("key2", b"a2345")
        self.assertTrue(checkpoint.exists("key1"))
        self.assertEqual(b"a1234", checkpoint.get("key1"))
        self.assertEqual(b"a2345", checkpoint.get("key2"))

    def test_server_checkpoint(self):
        checkpoint = ServerCheckpoint(
            remote_dir=f"/home/{self.admin_username}/test-dir",
            admin_username=self.admin_username,
            ip_address=self.ip_address,
        )
        self.assertFalse(checkpoint.exists("key1"))
        checkpoint.set("key1", b"b1234")
        checkpoint.set("key2", b"b2345")
        self.assertTrue(checkpoint.exists("key1"))
        self.assertEqual(b"b1234", checkpoint.get("key1"))
        self.assertEqual(b"b2345", checkpoint.get("key2"))
