# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import abc
import copy
import json
import os

import yaml
from kubernetes import client

from maro.cli.k8s.utils.k8s_details_reader import K8sDetailsReader
from maro.cli.k8s.utils.k8s_details_writer import K8sDetailsWriter
from maro.cli.k8s.utils.params import K8sPaths
from maro.cli.utils.deployment_validator import DeploymentValidator
from maro.cli.utils.name_creator import NameCreator
from maro.cli.utils.subprocess import Subprocess
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class K8sExecutor(abc.ABC):
    def __init__(self, cluster_details: dict):
        self.cluster_details = cluster_details

        # General configs
        self.cluster_name = self.cluster_details["name"]
        self.cluster_id = self.cluster_details["id"]

        # Init k8s_client env
        self.load_k8s_context()

    # maro k8s create

    @staticmethod
    def _init_redis():
        """Create the redis service in the k8s cluster.

        Returns:
            None.
        """
        with open(f"{K8sPaths.ABS_MARO_K8S_LIB}/configs/redis/redis.yml", "r") as fr:
            redis_deployment = yaml.safe_load(fr)
        client.AppsV1Api().create_namespaced_deployment(body=redis_deployment, namespace="default")

    @staticmethod
    @abc.abstractmethod
    def _init_nvidia_plugin():
        """ Init nvidia plugin for K8s Cluster.

        Different providers may have different loading mechanisms.

        Returns:
            None.
        """
        pass

    # maro k8s node

    def list_node(self):
        pass

    # maro k8s job

    def start_job(self, deployment_path: str) -> None:
        """Start a MARO Job with start_job_deployment.

        Args:
            deployment_path (str): path of the start_job_deployment.

        Returns:
            None.
        """
        # Load start_job_deployment.
        with open(deployment_path, "r") as fr:
            start_job_deployment = yaml.safe_load(fr)

        # Start job
        self._start_job(start_job_deployment=start_job_deployment)

    def _start_job(self, start_job_deployment: dict) -> None:
        """Start a MARO Job by converting the start_job_deployment to k8s job object and then execute it.

        Args:
            start_job_deployment (dict): raw start_job_deployment.

        Returns:
            None.
        """
        # Standardize start job deployment.
        job_details = K8sExecutor._standardize_job_details(start_job_deployment=start_job_deployment)

        # Save details
        K8sDetailsWriter.save_job_details(job_details=job_details)

        # Create and apply k8s config
        k8s_job = self._create_k8s_job(job_details=job_details)
        client.BatchV1Api().create_namespaced_job(body=k8s_job, namespace="default")

    @staticmethod
    def _standardize_job_details(start_job_deployment: dict) -> dict:
        """Standardize job_details with start_job_deployment.

        Args:
            start_job_deployment (dict): start_job_deployment of k8s/aks.
                See lib/deployments/internal for reference.

        Returns:
            dict: standardized job_details.
        """
        # Validate k8s_aks_start_job
        with open(f"{K8sPaths.ABS_MARO_K8S_LIB}/deployments/internal/k8s_aks_start_job.yml") as fr:
            start_job_template = yaml.safe_load(fr)
        DeploymentValidator.validate_and_fill_dict(
            template_dict=start_job_template,
            actual_dict=start_job_deployment,
            optional_key_to_value={}
        )

        # Validate component
        with open(f"{K8sPaths.ABS_MARO_K8S_LIB}/deployments/internal/component.yml", "r") as fr:
            component_template = yaml.safe_load(fr)
        components_details = start_job_deployment["components"]
        for _, component_details in components_details.items():
            DeploymentValidator.validate_and_fill_dict(
                template_dict=component_template,
                actual_dict=component_details,
                optional_key_to_value={}
            )

        # Init runtime fields.
        start_job_deployment["id"] = NameCreator.create_job_id()
        for component, component_details in start_job_deployment["components"].items():
            component_details["id"] = NameCreator.create_component_id()

        return start_job_deployment

    def _create_k8s_job(self, job_details: dict) -> dict:
        """Create k8s job object with job_details.

        Args:
            job_details (dict): details of the MARO Job.

        Returns:
            dict: k8s job object.
        """
        # Load details
        job_name = job_details["name"]
        job_id = job_details["id"]

        # Get config template
        with open(f"{K8sPaths.ABS_MARO_K8S_LIB}/configs/job/job.yml") as fr:
            k8s_job_config = yaml.safe_load(fr)
        with open(f"{K8sPaths.ABS_MARO_K8S_LIB}/configs/job/container.yml") as fr:
            k8s_container_config = yaml.safe_load(fr)

        # Fill configs
        k8s_job_config["metadata"]["name"] = job_id
        k8s_job_config["metadata"]["labels"]["jobName"] = job_name

        # Create and fill container config
        for component_type, component_details in job_details["components"].items():
            for component_index in range(component_details["num"]):
                k8s_job_config["spec"]["template"]["spec"]["containers"].append(
                    self._create_k8s_container_config(
                        job_details=job_details,
                        k8s_container_config_template=k8s_container_config,
                        component_type=component_type,
                        component_index=component_index
                    )
                )

        return k8s_job_config

    def _create_k8s_container_config(
        self,
        job_details: dict,
        k8s_container_config_template: dict,
        component_type: str,
        component_index: int
    ) -> dict:
        """Create the container config in the k8s job object.

        Args:
            job_details (dict): details of the MARO Job.
            k8s_container_config_template (dict): template of the k8s_container_config.
            component_type (str): type of the component.
            component_index (int): index of the component.

        Returns:
            dict: the container config.
        """
        # Copy config.
        k8s_container_config = copy.deepcopy(k8s_container_config_template)

        # Load details
        component_details = job_details["components"][component_type]
        job_id = job_details["id"]
        component_id = job_details["components"][component_type]["id"]
        container_name = f"{job_id}-{component_id}-{component_index}"

        # Fill configs.
        k8s_container_config["name"] = container_name
        k8s_container_config["image"] = component_details["image"]
        k8s_container_config["resources"]["requests"] = {
            "cpu": component_details["resources"]["cpu"],
            "memory": component_details["resources"]["memory"],
            "nvidia.com/gpu": component_details["resources"]["gpu"]
        }
        k8s_container_config["resources"]["limits"] = {
            "cpu": component_details["resources"]["cpu"],
            "memory": component_details["resources"]["memory"],
            "nvidia.com/gpu": component_details["resources"]["gpu"]
        }
        k8s_container_config["env"] = [
            {
                "name": "CLUSTER_ID",
                "value": f"{self.cluster_id}"
            },
            {
                "name": "CLUSTER_NAME",
                "value": f"{self.cluster_name}"
            },
            {
                "name": "JOB_ID",
                "value": f"{job_id}"
            },
            {
                "name": "JOB_NAME",
                "value": job_details["name"]
            },
            {
                "name": "COMPONENT_ID",
                "value": f"{component_id}"
            },
            {
                "name": "COMPONENT_TYPE",
                "value": f"{component_type}"
            },
            {
                "name": "COMPONENT_INDEX",
                "value": f"{component_index}"
            },
            {
                "name": "PYTHONUNBUFFERED",
                "value": "0"
            }
        ]
        k8s_container_config["command"] = component_details["command"]
        k8s_container_config["volumeMounts"][0]["mountPath"] = component_details["mount"]["target"]

        return k8s_container_config

    @staticmethod
    def stop_job(job_name: str) -> None:
        """Activate stop job operation.

        Args:
            job_name (str): name of the MARO Job.

        Returns:
            None.
        """
        K8sExecutor._stop_job(job_name=job_name)

    @staticmethod
    def _stop_job(job_name: str):
        """Stop MARO Job by stop k8s job object.

        Args:
            job_name (str): name of the MARO Job.

        Returns:
            None.
        """
        job_details = K8sDetailsReader.load_job_details(job_name=job_name)
        client.BatchV1Api().delete_namespaced_job(name=job_details["id"], namespace="default")

    @staticmethod
    def _export_log(pod_id: str, container_name: str, export_dir: str):
        """Export k8s job logs to the specific folder.

        Args:
            pod_id (str): id of the k8s pod.
            container_name (str): name of the container.
            export_dir (str): path of the exported folder.

        Returns:
            None.
        """
        os.makedirs(name=os.path.expanduser(export_dir + f"/{pod_id}"), exist_ok=True)
        with open(os.path.expanduser(export_dir + f"/{pod_id}/{container_name}.log"), "w") as fw:
            return_str = client.CoreV1Api().read_namespaced_pod_log(name=pod_id, namespace="default")
            fw.write(return_str)

    @staticmethod
    def list_job() -> None:
        """Print job_details of the cluster.

        Returns:
            None.
        """
        # Get jobs details
        job_list = client.BatchV1Api().list_namespaced_job(namespace="default").to_dict()["items"]

        # Print details
        logger.info(
            json.dumps(
                job_list,
                indent=4,
                sort_keys=True,
                default=str
            )
        )

    def get_job_logs(self, job_name: str, export_dir: str = "./") -> None:
        """Export MARO Job logs to the specific folder.

        Args:
            job_name (str): name of the MARO Job.
            export_dir (str): path of the exported folder.

        Returns:
            None.
        """
        # Load details
        job_details = K8sDetailsReader.load_job_details(job_name=job_name)

        # Get pods details
        pods_details = client.CoreV1Api().list_pod_for_all_namespaces().to_dict()["items"]

        # Reformat export_dir
        export_dir = os.path.expanduser(path=f"{export_dir}/{job_name}")

        # Export logs
        for pod_details in pods_details:
            if pod_details["metadata"]["name"].startswith(job_details["id"]):
                for container_details in pod_details["spec"]["containers"]:
                    self._export_log(
                        pod_id=pod_details["metadata"]["name"],
                        container_name=container_details["name"],
                        export_dir=export_dir
                    )

    # maro k8s schedule

    def start_schedule(self, deployment_path: str) -> None:
        """Start a MARO Schedule with start_schedule_deployment.

        Args:
            deployment_path (str): path of the start_schedule_deployment.

        Returns:
            None.
        """
        # Load start_schedule_deployment
        with open(deployment_path, "r") as fr:
            start_schedule_deployment = yaml.safe_load(fr)

        # Standardize start_schedule_deployment
        schedule_details = K8sExecutor._standardize_schedule_details(
            start_schedule_deployment=start_schedule_deployment
        )

        # Save schedule deployment
        K8sDetailsWriter.save_schedule_details(schedule_details=schedule_details)

        # Start jobs
        for job_name in schedule_details["job_names"]:
            job_details = K8sExecutor._build_job_details_for_schedule(
                schedule_details=schedule_details,
                job_name=job_name
            )
            self._start_job(start_job_deployment=job_details)

    def stop_schedule(self, schedule_name: str) -> None:
        """Stop a MARO Schedule.

        Args:
            schedule_name (str): name of the MARO Schedule.

        Returns:
            None.
        """
        schedule_details = K8sDetailsReader.load_schedule_details(schedule_name=schedule_name)
        job_names = schedule_details["job_names"]

        for job_name in job_names:
            # Load job details
            job_details = K8sDetailsReader.load_job_details(job_name=job_name)
            job_schedule_tag = job_details["tags"]["schedule_name"]

            # Stop job
            if job_schedule_tag == schedule_name:
                self._stop_job(job_name=job_name)

    @staticmethod
    def _standardize_schedule_details(start_schedule_deployment: dict) -> dict:
        """Standardize schedule_details with start_schedule_deployment.

        Args:
            start_schedule_deployment (dict): start_schedule_deployment of k8s/aks.
                See lib/deployments/internal for reference.

        Returns:
            dict: standardized job_details.
        """
        # Validate k8s_aks_start_schedule
        with open(f"{K8sPaths.ABS_MARO_K8S_LIB}/deployments/internal/k8s_aks_start_schedule.yml") as fr:
            start_job_template = yaml.safe_load(fr)
        DeploymentValidator.validate_and_fill_dict(
            template_dict=start_job_template,
            actual_dict=start_schedule_deployment,
            optional_key_to_value={}
        )

        # Validate component
        with open(f"{K8sPaths.ABS_MARO_K8S_LIB}/deployments/internal/component.yml") as fr:
            start_job_component_template = yaml.safe_load(fr)
        components_details = start_schedule_deployment["components"]
        for _, component_details in components_details.items():
            DeploymentValidator.validate_and_fill_dict(
                template_dict=start_job_component_template,
                actual_dict=component_details,
                optional_key_to_value={}
            )

        # Init runtime params
        start_schedule_deployment["id"] = NameCreator.create_schedule_id()

        return start_schedule_deployment

    @staticmethod
    def _build_job_details_for_schedule(schedule_details: dict, job_name: str) -> dict:
        """Build job_details from MARO Schedule.

        Args:
            schedule_details (dict): details of the MARO Schedule.
            job_name (str): name of the MARO Job.

        Returns:
            None.
        """
        # Convert schedule_details to job_details
        job_details = copy.deepcopy(schedule_details)
        job_details["name"] = job_name
        job_details["tags"] = {
            "schedule_name": schedule_details["name"],
            "schedule_id": schedule_details["id"]
        }
        job_details.pop("job_names")

        return job_details

    # maro k8s status

    @staticmethod
    def status():
        """Print details of specific MARO Resources (redis only at this time).

        Returns:
            None.
        """
        # Get resources
        pod_list = client.CoreV1Api().list_pod_for_all_namespaces(watch=False).to_dict()["items"]

        # Build return status
        return_status = {
            "redis": {
                "private_ip_address": K8sExecutor._get_redis_private_ip_address(pod_list=pod_list)
            }
        }

        # Print status
        logger.info(
            json.dumps(
                return_status,
                indent=4,
                sort_keys=True,
                default=str
            )
        )

    @staticmethod
    def _get_redis_private_ip_address(pod_list: list) -> str:
        """Get private_ip_address of the redis.

        Args:
            pod_list (list):

        Returns:
            str: private_ip_address.
        """
        for pod in pod_list:
            if "app" in pod["metadata"]["labels"] and pod["metadata"]["labels"]["app"] == "maro-redis":
                return pod["status"]["pod_ip"]
        return ""

    # maro k8s template

    @staticmethod
    def template(export_path: str) -> None:
        """Export deployment template of k8s mode.

        Args:
            export_path (str): location to export the templates.

        Returns:
            None.
        """
        command = f"cp {K8sPaths.ABS_MARO_K8S_LIB}/deployments/external/* {export_path}"
        _ = Subprocess.run(command=command)

    # Utils related

    @abc.abstractmethod
    def load_k8s_context(self):
        """ Load k8s context of the MARO cluster.

        Different providers have different loading mechanisms,
        but every override methods must invoke "config.load_kube_config()" at the very end.

        Returns:
            None.
        """
        pass
