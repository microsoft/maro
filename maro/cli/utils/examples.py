# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


MARO_GRASS_CREATE = """
Examples:
    Create a cluster in grass mode with a deployment
        maro grass create ./grass-create.yml
"""

MARO_GRASS_DELETE = """
Examples:
    Delete the cluster
        maro grass delete MyClusterName
"""

MARO_GRASS_NODE_SCALE = """
Examples:
    Scale the nodes with node size 'Standard_D4s_v3' to 2
        maro grass node scale MyClusterName Standard_D4s_v3 2

    Remove all nodes with node size 'Standard_D4s_v3'
        maro grass node scale MyClusterName Standard_D4s_v3 0

    For Azure, see more node sizes at https://docs.microsoft.com/en-us/azure/virtual-machines/sizes-general
"""

MARO_GRASS_NODE_START = """
Examples:
    Start 2 nodes with node size 'Standard_D4s_v3'
        maro grass node start MyClusterName Standard_D4s_v3 2
"""

MARO_GRASS_NODE_STOP = """
Examples:
    Stop 2 nodes with spec 'Standard_D4s_v3'
        maro grass node stop MyClusterName Standard_D4s_v3 2
"""

MARO_GRASS_NODE_LIST = """
Examples:
    List all nodes in the cluster
        maro grass node list MyClusterName

"""

MARO_GRASS_NODE_JOIN = """
Examples:
    Let one node join in a cluster in on-premises mode.
        maro grass node join node_join_info.yml
"""

MARO_GRASS_NODE_LEAVE = """
Examples:
    Let one node leave in a cluster in on-premises mode.
        maro grass node leave MyClusterName node_ip_address
"""

MARO_GRASS_IMAGE_PUSH = """
Examples:
    Push a local image in the docker registry and load it into cluster
        maro grass image push MyClusterName --image-name MyImage

    Push a local image file and load it into cluster
        maro grass image push MyClusterName --image-path ./image.tar

    Build an image based on the remote context folder and load it into cluster
        maro grass image push MyClusterName --remote-context-path /image_context --remote-name MyImageName
"""

MARO_GRASS_DATA_PUSH = """
Examples:
    Push the local data to the remote directory
        maro grass data push MyClusterName ./my_local_data/* /remote_data_folder
"""

MARO_GRASS_DATA_PULL = """
Examples:
    Pull the remote data to local directory
        maro grass data pull MyClusterName /remote_data_folder/* ./my_local_data/
"""

MARO_GRASS_JOB_START = """
Examples:
    Start a training job with a deployment
        maro grass job start MyClusterName ./grass-start-job.yml
"""

MARO_GRASS_JOB_STOP = """
Examples:
    Stop a training job
        maro grass job stop MyClusterName MyJobName
"""

MARO_GRASS_JOB_LIST = """
Examples:
    List all jobs in the cluster
        maro grass job list MyClusterName
"""

MARO_GRASS_JOB_LOGS = """
Examples:
    Get logs of the job to current directory
        maro grass job logs MyClusterName MyJobName
"""

MARO_GRASS_SCHEDULE_START = """
Examples:
    Start a training schedule with a deployment
        maro grass job start MyClusterName ./grass-start-schedule.yml
"""

MARO_GRASS_SCHEDULE_STOP = """
Examples:
    Stop a training schedule
        maro grass job stop MyClusterName MyScheduleName
"""

MARO_GRASS_CLEAN = """
Examples:
    Clean the cluster
        maro grass clean MyClusterName
"""

MARO_GRASS_STATUS = """
Examples:
    Get status of the resource in the cluster
        maro grass status MyClusterName master
        maro grass status MyClusterName nodes
"""

MARO_GRASS_TEMPLATES = """
Examples:
    Get deployment templates to target directory
        maro grass template
"""

MARO_K8S_CREATE = """
Examples:
    Create a cluster in k8s mode with a deployment
        maro k8s create ./k8s-create.yml
"""

MARO_K8S_DELETE = """
Examples:
    Delete the cluster
        maro k8s delete MyClusterName
"""

MARO_K8S_NODE_SCALE = """
Examples:
    Scale the nodes with node size 'Standard_D4s_v3' to 2
        maro k8s node scale MyClusterName Standard_D4s_v3 2

    Remove all nodes with node size 'Standard_D4s_v3'
        maro k8s node scale MyClusterName Standard_D4s_v3 0

    For Azure, see more node sizes at https://docs.microsoft.com/en-us/azure/virtual-machines/sizes-general
"""

MARO_K8S_NODE_LIST = """
Examples:
    List all nodes in the cluster
        maro k8s node list MyClusterName
"""

MARO_K8S_IMAGE_PUSH = """
Examples:
    Push a local image in the docker registry and load it into cluster
        maro k8s image push MyClusterName --image-name MyImage
"""

MARO_K8S_IMAGE_LIST = """
Examples:
    List all images in the cluster
        maro k8s image list MyClusterName
"""

MARO_K8S_DATA_PUSH = """
Examples:
    Push the local data to the remote directory
        maro k8s data push MyClusterName ./my_local_data/* /remote_data_folder
"""

MARO_K8S_DATA_PULL = """
Examples:
    Pull the remote data to local directory
        maro k8s data pull MyClusterName /remote_data_folder/* ./my_local_data/
"""

MARO_K8S_JOB_START = """
Examples:
    Start a training job with a deployment
        maro k8s job start MyClusterName ./k8s-start-job.yml
"""

MARO_K8S_JOB_STOP = """
Examples:
    Stop a training job
        maro k8s job stop MyClusterName MyJobName
"""

MARO_K8S_JOB_LIST = """
Examples:
    List all jobs in the cluster
        maro k8s job list MyClusterName
"""

MARO_K8S_JOB_LOGS = """
Examples:
    Get logs of the job to current directory
        maro k8s job logs MyClusterName MyJobName
"""

MARO_K8S_SCHEDULE_START = """
Examples:
    Start a training schedule with a deployment
        maro k8s job start MyClusterName ./k8s-start-schedule.yml
"""

MARO_K8S_SCHEDULE_STOP = """
Examples:
    Stop a training schedule
        maro k8s job stop MyClusterName MyScheduleName
"""

MARO_K8S_STATUS = """
Examples:
    Get status of the cluster
        maro k8s status MyClusterName
"""

MARO_K8S_TEMPLATE = """
Examples:
    Get deployment templates to target directory
        maro k8s template

"""

MARO_PROCESS_SETUP = """
Examples:
    Start Redis and agents for local process mode.
        maro process setup [setting.yml]
"""

MARO_PROCESS_JOB_START = """
Examples:
    Start a training job with a deployment
        maro process job start ./process-start-job.yml
"""

MARO_PROCESS_JOB_STOP = """
Examples:
    Stop a training job with a job name
        maro process job stop job_name
"""

MARO_PROCESS_JOB_DELETE = """
Examples:
    Delete local job files with a job name
        maro process job delete job_name
"""

MARO_PROCESS_JOB_LIST = """
Examples:
    List all jobs
        maro process job list
"""

MARO_PROCESS_JOB_LOGS = """
Examples:
    Get logs of the job to current directory
        maro process job logs job_name
"""

MARO_PROCESS_SCHEDULE_START = """
Examples:
    Start a schedule with a deployment
        maro process schedule start ./process-start-schedule.yml
"""

MARO_PROCESS_SCHEDULE_STOP = """
Examples:
    Stop a training schedule with a schedule name
        maro process schedule stop schedule_name
"""

MARO_PROCESS_TEMPLATE = """
Examples:
    Get deployment templates (include setting template) to target directory
        maro k8s template --setting_deploy ./target_directory
"""
