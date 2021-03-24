# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import logging
import traceback
from argparse import Namespace
from copy import deepcopy

import maro.cli.utils.examples as CliExamples
from maro import __version__
from maro.cli.utils.params import GlobalParams
from maro.cli.utils.parser import ArgumentParser
from maro.utils.exception.cli_exception import CliError, CommandNotFoundError
from maro.utils.logger import CliLogger

MARO_BANNER = """
___  __    _    ____   ___
|  \/  |  / \  |  _ \ / _ \\
| |\/| | / _ \ | |_) | | | |
| |  | |/ ___ \|  _ <| |_| |
|_|  |_/_/   \_\_| \_ \___/

Welcome to the MARO CLI

Use `maro --version` to get the current version.

"""

logger = CliLogger(name=__name__)


def main():
    global_parser = ArgumentParser()
    global_parser.add_argument("--debug", action='store_true', help="Enable debug mode")
    global_parser.add_argument("-h", "--help", action='store_true', help="Show this message and exit")

    parser = ArgumentParser(prog='maro', description=MARO_BANNER, parents=[global_parser])
    parser.set_defaults(func=_help_func(parser=parser))
    parser.add_argument('--version', action='store_true', help='Get version info')
    subparsers = parser.add_subparsers()

    # maro env
    parser_env = subparsers.add_parser(
        'env',
        help=(
            'Get all environment-related information, '
            'such as the supported scenarios, topologies. \n'
            'And it is also responsible to generate data to the specific environment, '
            'which has external data dependency.'
        ),
        parents=[global_parser]
    )
    parser_env.set_defaults(func=_help_func(parser=parser_env))
    load_parser_env(parser_env, global_parser)

    # maro data
    parser_data = subparsers.add_parser(
        'data',
        help="Data processing tools for MARO binary format.",
        parents=[global_parser]
    )
    parser_data.set_defaults(func=_help_func(parser=parser_data))
    load_parser_data(prev_parser=parser_data, global_parser=global_parser)

    # maro meta
    parser_meta = subparsers.add_parser(
        'meta',
        help="Manage the meta files for MARO.",
        parents=[global_parser]
    )
    parser_meta.set_defaults(func=_help_func(parser=parser_meta))
    load_parser_meta(prev_parser=parser_meta, global_parser=global_parser)

    # maro grass
    parser_grass = subparsers.add_parser(
        "grass",
        help="Manage distributed cluster with native virtual machines (for development only).",
        parents=[global_parser]
    )
    parser_grass.set_defaults(func=_help_func(parser=parser_grass))
    load_parser_grass(prev_parser=parser_grass, global_parser=global_parser)

    # maro k8s
    parser_k8s = subparsers.add_parser(
        "k8s",
        help="Manage distributed cluster with Kubernetes.",
        parents=[global_parser]
    )
    parser_k8s.set_defaults(func=_help_func(parser=parser_k8s))
    load_parser_k8s(prev_parser=parser_k8s, global_parser=global_parser)

    # maro inspector
    parser_inspector = subparsers.add_parser(
        'inspector',
        help=("Display visualization of experimental data."),
        parents=[global_parser]
    )
    parser_inspector.set_defaults(func=_help_func(parser=parser_inspector))
    load_parser_inspector(parser_inspector, global_parser)

    # maro process
    parser_process = subparsers.add_parser(
        "process",
        help="Run application by mulit-process to simulate distributed mode."
    )
    parser_process.set_defaults(func=_help_func(parser=parser_process))
    load_parser_process(prev_parser=parser_process, global_parser=global_parser)

    # maro project
    parser_project = subparsers.add_parser(
        "project",
        help="Manage maro projects."
    )

    parser_project.set_defaults(func=_help_func(parser=load_parser_project))
    load_parser_project(prev_parser=parser_project, global_parser=global_parser)

    # maro admin
    parser_admin = subparsers.add_parser(
        "admin",
        help="Manage maro admin tools."
    )

    parser_admin.set_defaults(func=_help_func(parser=parser_admin))
    load_parser_admin(prev_parser=parser_admin, global_parser=global_parser)

    args = None
    try:
        # Get args and parse global arguments
        args = parser.parse_args()
        if args.debug:
            GlobalParams.LOG_LEVEL = logging.DEBUG
        else:
            GlobalParams.LOG_LEVEL = logging.INFO
        if args.version:
            logger.info(f'{__version__}')
            return

        actual_args = _get_actual_args(namespace=args)

        # WARNING: We cannot assign any argument like 'func' in the CLI
        args.func(**actual_args)
    except CommandNotFoundError as e:
        logger.error_red(f"{e.__class__.__name__}: {e.get_message()}")
        logger.info(f"{e.usage}")
    except CliError as e:
        if args is None or args.debug:
            logger.error_red(f"{e.__class__.__name__}: {e.get_message()}\n{traceback.format_exc()}")
        else:
            logger.error_red(f"{e.__class__.__name__}: {e.get_message()}")


def load_parser_process(prev_parser: ArgumentParser, global_parser: ArgumentParser) -> None:
    subparsers = prev_parser.add_subparsers()

    # maro process create
    from maro.cli.process.create import create
    parser_setup = subparsers.add_parser(
        "create",
        help="Create local process environment.",
        examples=CliExamples.MARO_PROCESS_SETUP,
        parents=[global_parser]
    )
    parser_setup.add_argument(
        'deployment_path',
        help='Path of the local process setting deployment.',
        nargs='?',
        default=None)
    parser_setup.set_defaults(func=create)

    # maro process delete
    from maro.cli.process.delete import delete
    parser_setup = subparsers.add_parser(
        "delete",
        help="Delete the local process environment. Including closing agents and maro Redis.",
        parents=[global_parser]
    )
    parser_setup.set_defaults(func=delete)

    # maro process job
    parser_job = subparsers.add_parser(
        "job",
        help="Manage jobs",
        parents=[global_parser]
    )
    parser_job.set_defaults(func=_help_func(parser=parser_job))
    parser_job_subparsers = parser_job.add_subparsers()

    # maro process job start
    from maro.cli.process.job import start_job
    parser_job_start = parser_job_subparsers.add_parser(
        'start',
        help='Start a training job',
        examples=CliExamples.MARO_PROCESS_JOB_START,
        parents=[global_parser]
    )
    parser_job_start.add_argument(
        'deployment_path', help='Path of the job deployment')
    parser_job_start.set_defaults(func=start_job)

    # maro process job stop
    from maro.cli.process.job import stop_job
    parser_job_stop = parser_job_subparsers.add_parser(
        'stop',
        help='Stop a training job',
        examples=CliExamples.MARO_PROCESS_JOB_STOP,
        parents=[global_parser]
    )
    parser_job_stop.add_argument(
        'job_name', help='Name of the job')
    parser_job_stop.set_defaults(func=stop_job)

    # maro process job delete
    from maro.cli.process.job import delete_job
    parser_job_delete = parser_job_subparsers.add_parser(
        'delete',
        help='delete a stopped job',
        examples=CliExamples.MARO_PROCESS_JOB_DELETE,
        parents=[global_parser]
    )
    parser_job_delete.add_argument(
        'job_name', help='Name of the job or the schedule')
    parser_job_delete.set_defaults(func=delete_job)

    # maro process job list
    from maro.cli.process.job import list_jobs
    parser_job_list = parser_job_subparsers.add_parser(
        'list',
        help='List all jobs',
        examples=CliExamples.MARO_PROCESS_JOB_LIST,
        parents=[global_parser]
    )
    parser_job_list.set_defaults(func=list_jobs)

    # maro process job logs
    from maro.cli.process.job import get_job_logs
    parser_job_logs = parser_job_subparsers.add_parser(
        'logs',
        help='Get logs of the job',
        examples=CliExamples.MARO_PROCESS_JOB_LOGS,
        parents=[global_parser]
    )
    parser_job_logs.add_argument(
        'job_name', help='Name of the job')
    parser_job_logs.set_defaults(func=get_job_logs)

    # maro process schedule
    parser_schedule = subparsers.add_parser(
        'schedule',
        help='Manage schedules',
        parents=[global_parser]
    )
    parser_schedule.set_defaults(func=_help_func(parser=parser_schedule))
    parser_schedule_subparsers = parser_schedule.add_subparsers()

    # maro process schedule start
    from maro.cli.process.schedule import start_schedule
    parser_schedule_start = parser_schedule_subparsers.add_parser(
        'start',
        help='Start a schedule',
        examples=CliExamples.MARO_PROCESS_SCHEDULE_START,
        parents=[global_parser]
    )
    parser_schedule_start.add_argument(
        'deployment_path', help='Path of the schedule deployment')
    parser_schedule_start.set_defaults(func=start_schedule)

    # maro process schedule stop
    from maro.cli.process.schedule import stop_schedule
    parser_schedule_stop = parser_schedule_subparsers.add_parser(
        'stop',
        help='Stop a schedule',
        examples=CliExamples.MARO_PROCESS_SCHEDULE_STOP,
        parents=[global_parser]
    )
    parser_schedule_stop.add_argument(
        'schedule_name', help='Name of the schedule')
    parser_schedule_stop.set_defaults(func=stop_schedule)

    # maro process template
    from maro.cli.process.template import template
    parser_template = subparsers.add_parser(
        "template",
        help="Get deployment templates",
        examples=CliExamples.MARO_PROCESS_TEMPLATE,
        parents=[global_parser]
    )
    parser_template.add_argument(
        "--setting_deploy",
        action="store_true",
        help="Get environment setting templates"
    )
    parser_template.add_argument(
        "export_path",
        default="./",
        nargs='?',
        help="Path of the export directory")
    parser_template.set_defaults(func=template)


def load_parser_grass(prev_parser: ArgumentParser, global_parser: ArgumentParser) -> None:
    subparsers = prev_parser.add_subparsers()

    # maro grass create
    from maro.cli.grass.create import create
    parser_create = subparsers.add_parser(
        "create",
        help="Create cluster",
        examples=CliExamples.MARO_GRASS_CREATE,
        parents=[global_parser]
    )
    parser_create.add_argument("deployment_path", help="Path of the create deployment")
    parser_create.set_defaults(func=create)

    # maro grass delete
    from maro.cli.grass.delete import delete
    parser_delete = subparsers.add_parser(
        "delete",
        help="Delete cluster",
        examples=CliExamples.MARO_GRASS_DELETE,
        parents=[global_parser]
    )
    parser_delete.add_argument("cluster_name", help="Name of the cluster")
    parser_delete.set_defaults(func=delete)

    # maro grass node
    parser_node = subparsers.add_parser(
        "node",
        help="Manage nodes of the cluster",
        parents=[global_parser]
    )
    parser_node.set_defaults(func=_help_func(parser=parser_node))
    parser_node_subparsers = parser_node.add_subparsers()

    # maro grass node scale
    from maro.cli.grass.node import scale_node
    parser_node_scale = parser_node_subparsers.add_parser(
        "scale",
        help="Scale up or scale down nodes to target number",
        examples=CliExamples.MARO_GRASS_NODE_SCALE,
        parents=[global_parser]
    )
    parser_node_scale.add_argument("cluster_name", help="Name of the cluster")
    parser_node_scale.add_argument("node_size", help="Azure VM size")
    parser_node_scale.add_argument("replicas", type=int, help="Target number of the nodes in the specific node_size")
    parser_node_scale.set_defaults(func=scale_node)

    # maro grass node start
    from maro.cli.grass.node import start_node
    parser_node_start = parser_node_subparsers.add_parser(
        "start",
        help="Start nodes",
        examples=CliExamples.MARO_GRASS_NODE_START,
        parents=[global_parser]
    )
    parser_node_start.add_argument("cluster_name", help="Name of the cluster")
    parser_node_start.add_argument("node_size", help="Azure VM size")
    parser_node_start.add_argument(
        "replicas",
        type=int,
        help="Target number of the nodes need to be started in the specific node_size"
    )
    parser_node_start.set_defaults(func=start_node)

    # maro grass node stop
    from maro.cli.grass.node import stop_node
    parser_node_stop = parser_node_subparsers.add_parser(
        "stop",
        help="Stop nodes",
        examples=CliExamples.MARO_GRASS_NODE_STOP,
        parents=[global_parser]
    )
    parser_node_stop.add_argument("cluster_name", help="Name of the cluster")
    parser_node_stop.add_argument("node_size", help="Azure VM size")
    parser_node_stop.add_argument(
        "replicas",
        type=int,
        help="Target number of the nodes need to be stopped in the specific node_size"
    )
    parser_node_stop.set_defaults(func=stop_node)

    # maro grass node list
    from maro.cli.grass.node import list_node
    parser_node_list = parser_node_subparsers.add_parser(
        "list",
        help="List details of nodes",
        examples=CliExamples.MARO_GRASS_NODE_LIST,
        parents=[global_parser]
    )
    parser_node_list.add_argument("cluster_name", help="Name of the cluster")
    parser_node_list.set_defaults(func=list_node)

    # maro grass node join
    from maro.cli.grass.node import join_cluster
    parser_node_join = parser_node_subparsers.add_parser(
        "join",
        help="Let one node join in a cluster in on-premises mode.",
        examples=CliExamples.MARO_GRASS_NODE_JOIN,
        parents=[global_parser]
    )
    parser_node_join.add_argument("deployment_path", help="The node join description file path.")
    parser_node_join.set_defaults(func=join_cluster)

    # maro grass node leave
    from maro.cli.grass.node import leave_cluster
    parser_node_leave = parser_node_subparsers.add_parser(
        "leave",
        help="make node leave to cluster",
        examples=CliExamples.MARO_GRASS_NODE_LEAVE,
        parents=[global_parser]
    )
    parser_node_leave.add_argument(
        "deployment_path",
        nargs="?",
        default={},
        help="The node join description file path."
    )
    parser_node_leave.set_defaults(func=leave_cluster)

    # maro grass image
    parser_image = subparsers.add_parser(
        "image",
        help="Manage images of the cluster",
        parents=[global_parser]
    )
    parser_image.set_defaults(func=_help_func(parser=parser_image))
    parser_image_subparsers = parser_image.add_subparsers()

    # maro grass image push
    from maro.cli.grass.image import push_image
    parser_image_push = parser_image_subparsers.add_parser(
        "push",
        help="Push a local image to the cluster",
        examples=CliExamples.MARO_GRASS_IMAGE_PUSH,
        parents=[global_parser]
    )
    parser_image_push.add_argument("cluster_name", help="Name of the cluster")
    parser_image_push.add_argument("--image-name", help="Name of the local image")
    parser_image_push.add_argument("--image-path", help="Path of the local tar file")
    parser_image_push.add_argument(
        "--remote-context-path",
        help="Absolute path of the image context in the user data storage of the cluster"
    )
    parser_image_push.add_argument("--remote-image-name", help="Name of the image")
    parser_image_push.set_defaults(func=push_image)

    # maro grass data
    parser_data = subparsers.add_parser(
        "data",
        help="Manage user data storage in the cluster",
        parents=[global_parser]
    )
    parser_data.set_defaults(func=_help_func(parser=parser_data))
    parser_data_subparsers = parser_data.add_subparsers()

    # maro grass data push
    from maro.cli.grass.data import push_data
    parser_data_push = parser_data_subparsers.add_parser(
        "push",
        help="Push the local data to the remote directory",
        examples=CliExamples.MARO_GRASS_DATA_PUSH,
        parents=[global_parser]
    )
    parser_data_push.add_argument("cluster_name", help="Name of the cluster")
    parser_data_push.add_argument("local_path", help="Path of the local file")
    parser_data_push.add_argument("remote_path", help="Path of the directory in the cluster data storage")
    parser_data_push.set_defaults(func=push_data)

    # maro grass data pull
    from maro.cli.grass.data import pull_data
    parser_data_pull = parser_data_subparsers.add_parser(
        "pull",
        help="Pull the remote data to the local directory",
        examples=CliExamples.MARO_GRASS_DATA_PULL,
        parents=[global_parser]
    )
    parser_data_pull.add_argument("cluster_name", help="Name of the cluster")
    parser_data_pull.add_argument("remote_path", help="Path of the file in the cluster data storage")
    parser_data_pull.add_argument("local_path", help="Path of the directory in the local")
    parser_data_pull.set_defaults(func=pull_data)

    # maro grass job
    parser_job = subparsers.add_parser(
        "job",
        help="Manage jobs",
        parents=[global_parser]
    )
    parser_job.set_defaults(func=_help_func(parser=parser_job))
    parser_job_subparsers = parser_job.add_subparsers()

    # maro grass job start
    from maro.cli.grass.job import start_job
    parser_job_start = parser_job_subparsers.add_parser(
        "start",
        help="Start a training job",
        examples=CliExamples.MARO_GRASS_JOB_START,
        parents=[global_parser]
    )
    parser_job_start.add_argument("cluster_name", help="Name of the cluster")
    parser_job_start.add_argument("deployment_path", help="Path of the job deployment")
    parser_job_start.set_defaults(func=start_job)

    # maro grass job stop
    from maro.cli.grass.job import stop_job
    parser_job_stop = parser_job_subparsers.add_parser(
        "stop",
        help="Stop a training job",
        examples=CliExamples.MARO_GRASS_JOB_STOP,
        parents=[global_parser]
    )
    parser_job_stop.add_argument("cluster_name", help="Name of the cluster")
    parser_job_stop.add_argument("job_name", help="Name of the job")
    parser_job_stop.set_defaults(func=stop_job)

    # maro grass job list
    from maro.cli.grass.job import list_job
    parser_job_list = parser_job_subparsers.add_parser(
        "list",
        help="List details of jobs",
        examples=CliExamples.MARO_GRASS_JOB_LIST,
        parents=[global_parser]
    )
    parser_job_list.add_argument("cluster_name", help="Name of the cluster")
    parser_job_list.set_defaults(func=list_job)

    # maro grass job logs
    from maro.cli.grass.job import get_job_logs
    parser_job_logs = parser_job_subparsers.add_parser(
        "logs",
        help="Get logs of the job",
        examples=CliExamples.MARO_GRASS_JOB_LOGS,
        parents=[global_parser]
    )
    parser_job_logs.add_argument("cluster_name", help="Name of the cluster")
    parser_job_logs.add_argument("job_name", help="Name of the job")
    parser_job_logs.set_defaults(func=get_job_logs)

    # maro grass schedule
    parser_schedule = subparsers.add_parser(
        "schedule",
        help="Manage schedules",
        parents=[global_parser]
    )
    parser_schedule.set_defaults(func=_help_func(parser=parser_schedule))
    parser_schedule_subparsers = parser_schedule.add_subparsers()

    # maro grass schedule start
    from maro.cli.grass.schedule import start_schedule
    parser_schedule_start = parser_schedule_subparsers.add_parser(
        "start",
        help="Start a schedule",
        examples=CliExamples.MARO_GRASS_SCHEDULE_START,
        parents=[global_parser]
    )
    parser_schedule_start.add_argument("cluster_name", help="Name of the cluster")
    parser_schedule_start.add_argument("deployment_path", help="Path of the schedule deployment")
    parser_schedule_start.set_defaults(func=start_schedule)

    # maro grass schedule stop
    from maro.cli.grass.schedule import stop_schedule
    parser_schedule_stop = parser_schedule_subparsers.add_parser(
        "stop",
        help="Stop a schedule",
        examples=CliExamples.MARO_GRASS_SCHEDULE_STOP,
        parents=[global_parser]
    )
    parser_schedule_stop.add_argument("cluster_name", help="Name of the cluster")
    parser_schedule_stop.add_argument("schedule_name", help="Name of the schedule")
    parser_schedule_stop.set_defaults(func=stop_schedule)

    # maro grass clean
    from maro.cli.grass.clean import clean
    parser_clean = subparsers.add_parser(
        "clean",
        help="Clean cluster",
        examples=CliExamples.MARO_GRASS_CLEAN,
        parents=[global_parser]
    )
    parser_clean.add_argument("cluster_name", help="Name of the cluster")
    parser_clean.set_defaults(func=clean)

    # maro grass status
    from maro.cli.grass.status import status
    parser_status = subparsers.add_parser(
        "status",
        help="Get status of the cluster",
        examples=CliExamples.MARO_GRASS_STATUS,
        parents=[global_parser]
    )
    parser_status.add_argument("cluster_name", help="Name of the cluster")
    parser_status.add_argument("resource_name", help="Name of the resource")
    parser_status.set_defaults(func=status)

    # maro grass template
    from maro.cli.grass.template import template
    parser_clean = subparsers.add_parser(
        "template",
        help="Get deployment templates",
        examples=CliExamples.MARO_GRASS_TEMPLATES,
        parents=[global_parser]
    )
    parser_clean.add_argument("export_path", default="./", help="Path of the export directory")
    parser_clean.set_defaults(func=template)


def load_parser_env(prev_parser: ArgumentParser, global_parser) -> None:
    from maro.cli.envs.list_available import list_scenarios, list_topologies

    subparsers = prev_parser.add_subparsers()

    # maro env list
    parser_list = subparsers.add_parser(
        'list',
        help='List name of built-in scenarios.',
        parents=[global_parser]
    )

    parser_list.set_defaults(func=list_scenarios)

    # maro env topologies --scenario name
    parser_topo = subparsers.add_parser(
        'topology',
        help='Get built-in topologies for specified scenario.',
        parents=[global_parser]
    )

    parser_topo.add_argument(
        '-s', '--scenario',
        required=True,
        help='Scenario name to show topologies.'
    )

    parser_topo.set_defaults(func=list_topologies)

    # MARO env data command
    parser_env_data = subparsers.add_parser(
        'data',
        help="Generate predefined scenario related data.",
        parents=[global_parser]
    )
    parser_env_data.set_defaults(func=_help_func(parser=parser_env_data))

    # Generate data for a specific scenario and topology.
    from maro.cli.data_pipeline.data_process import generate, list_env
    from maro.simulator.utils.common import get_scenarios
    data_subparsers = parser_env_data.add_subparsers()

    generate_cmd_parser = data_subparsers.add_parser(
        "generate",
        help="Generate data for a specific scenario and topology.",
        parents=[global_parser])

    generate_cmd_parser.add_argument(
        "-s", "--scenario",
        required=True,
        choices=get_scenarios(),
        help="Scenario of environment.")

    generate_cmd_parser.add_argument(
        "-t", "--topology",
        required=True,
        help="Topology of scenario.")

    generate_cmd_parser.add_argument(
        "-f", "--forced",
        action="store_true",
        help="Re-generate forcibly.")
    generate_cmd_parser.set_defaults(func=generate)

    list_cmd_parser = data_subparsers.add_parser(
        "list",
        help="List predefined environments that need generate data extraly.",
        parents=[global_parser])

    list_cmd_parser.set_defaults(func=list_env)


def load_parser_k8s(prev_parser: ArgumentParser, global_parser: ArgumentParser) -> None:
    subparsers = prev_parser.add_subparsers()

    # maro k8s create
    from maro.cli.k8s.create import create
    parser_create = subparsers.add_parser(
        "create",
        help="Create cluster",
        examples=CliExamples.MARO_K8S_CREATE,
        parents=[global_parser]
    )
    parser_create.add_argument("deployment_path", help="Path of the create deployment")
    parser_create.set_defaults(func=create)

    # maro k8s delete
    from maro.cli.k8s.delete import delete
    parser_create = subparsers.add_parser(
        "delete",
        help="Delete cluster",
        examples=CliExamples.MARO_K8S_DELETE,
        parents=[global_parser]
    )
    parser_create.add_argument("cluster_name", help="Name of the cluster")
    parser_create.set_defaults(func=delete)

    # maro k8s node
    parser_node = subparsers.add_parser(
        "node",
        help="Manage nodes of the cluster",
        parents=[global_parser]
    )
    parser_node.set_defaults(func=_help_func(parser=parser_node))
    parser_node_subparsers = parser_node.add_subparsers()

    # maro k8s node scale
    from maro.cli.k8s.node import scale_node
    parser_node_scale = parser_node_subparsers.add_parser(
        "scale",
        help="Scale up or scale down nodes to target number",
        examples=CliExamples.MARO_K8S_NODE_SCALE,
        parents=[global_parser]
    )
    parser_node_scale.add_argument("cluster_name", help="Name of the cluster")
    parser_node_scale.add_argument("node_size", help="Azure VM size")
    parser_node_scale.add_argument("replicas", type=int, help="Target number of the nodes in the specific node_size")
    parser_node_scale.set_defaults(func=scale_node)

    # maro k8s node list
    from maro.cli.k8s.node import list_node
    parser_node_scale = parser_node_subparsers.add_parser(
        "list",
        help="List details of nodes",
        examples=CliExamples.MARO_K8S_NODE_LIST,
        parents=[global_parser]
    )
    parser_node_scale.add_argument("cluster_name", help="Name of the cluster")
    parser_node_scale.set_defaults(func=list_node)

    # maro k8s image
    parser_image = subparsers.add_parser(
        "image",
        help="Manage images of the cluster",
        parents=[global_parser]
    )
    parser_image.set_defaults(func=_help_func(parser=parser_image))
    parser_image_subparsers = parser_image.add_subparsers()

    # maro k8s image push
    from maro.cli.k8s.image import push_image
    parser_image_push = parser_image_subparsers.add_parser(
        "push",
        help="Push a local image to the cluster",
        examples=CliExamples.MARO_K8S_IMAGE_PUSH,
        parents=[global_parser]
    )
    parser_image_push.add_argument("cluster_name", help="Name of the cluster")
    parser_image_push.add_argument("--image-name", help="Name of the local image")
    parser_image_push.set_defaults(func=push_image)

    # maro k8s image list
    from maro.cli.k8s.image import list_image
    parser_image_push = parser_image_subparsers.add_parser(
        "list",
        help="List the images in the cluster",
        examples=CliExamples.MARO_K8S_IMAGE_LIST,
        parents=[global_parser]
    )
    parser_image_push.add_argument("cluster_name", help="Name of the cluster")
    parser_image_push.set_defaults(func=list_image)

    # maro k8s data
    parser_data = subparsers.add_parser(
        "data",
        help="Manage user data storage in the cluster",
        parents=[global_parser]
    )
    parser_data.set_defaults(func=_help_func(parser=parser_data))
    parser_data_subparsers = parser_data.add_subparsers()

    # maro k8s data push
    from maro.cli.k8s.data import push_data
    parser_data_push = parser_data_subparsers.add_parser(
        "push",
        help="Push the local data to the remote directory",
        examples=CliExamples.MARO_K8S_DATA_PUSH,
        parents=[global_parser]
    )
    parser_data_push.add_argument("cluster_name", help="Name of the cluster")
    parser_data_push.add_argument("local_path", help="Path of the local file")
    parser_data_push.add_argument("remote_dir", help="Path of the directory in the cluster data storage")
    parser_data_push.set_defaults(func=push_data)

    # maro k8s data pull
    from maro.cli.k8s.data import pull_data
    parser_data_pull = parser_data_subparsers.add_parser(
        "pull",
        help="Pull the remote data to the local directory",
        examples=CliExamples.MARO_K8S_DATA_PULL,
        parents=[global_parser]
    )
    parser_data_pull.add_argument("cluster_name", help="Name of the cluster")
    parser_data_pull.add_argument("remote_path", help="Path of the file in the cluster data storage")
    parser_data_pull.add_argument("local_dir", help="Path of the directory in the local")
    parser_data_pull.set_defaults(func=pull_data)

    # maro k8s data remove
    from maro.cli.k8s.data import remove_data
    parser_data_pull = parser_data_subparsers.add_parser(
        "remove",
        help="Remove data in the cluster data storage",
        parents=[global_parser]
    )
    parser_data_pull.add_argument("cluster_name", help="Name of the cluster")
    parser_data_pull.add_argument("remote_path", help="Path of the file in the cluster data storage")
    parser_data_pull.set_defaults(func=remove_data)

    # maro k8s job
    parser_job = subparsers.add_parser(
        "job",
        help="Manage jobs",
        parents=[global_parser]
    )
    parser_job.set_defaults(func=_help_func(parser=parser_job))
    parser_job_subparsers = parser_job.add_subparsers()

    # maro k8s job start
    from maro.cli.k8s.job import start_job
    parser_job_start = parser_job_subparsers.add_parser(
        "start",
        help="Start a training job",
        examples=CliExamples.MARO_K8S_JOB_START,
        parents=[global_parser]
    )
    parser_job_start.add_argument("cluster_name", help="Name of the cluster")
    parser_job_start.add_argument("deployment_path", help="Path of the job deployment")
    parser_job_start.set_defaults(func=start_job)

    # maro k8s job stop
    from maro.cli.k8s.job import stop_job
    parser_job_stop = parser_job_subparsers.add_parser(
        "stop",
        help="Stop a training job",
        examples=CliExamples.MARO_K8S_JOB_STOP,
        parents=[global_parser]
    )
    parser_job_stop.add_argument("cluster_name", help="Name of the cluster")
    parser_job_stop.add_argument("job_name", help="Name of the job")
    parser_job_stop.set_defaults(func=stop_job)

    # maro k8s job logs
    from maro.cli.k8s.job import get_job_logs
    parser_job_logs = parser_job_subparsers.add_parser(
        "logs",
        help="Get logs of the job",
        parents=[global_parser]
    )
    parser_job_logs.add_argument("cluster_name", help="Name of the cluster")
    parser_job_logs.add_argument("job_name", help="Name of the job")
    parser_job_logs.set_defaults(func=get_job_logs)

    # maro k8s job list
    from maro.cli.k8s.job import list_job
    parser_job_list = parser_job_subparsers.add_parser(
        "list",
        help="List details of jobs",
        parents=[global_parser]
    )
    parser_job_list.add_argument("cluster_name", help="Name of the cluster")
    parser_job_list.set_defaults(func=list_job)

    # maro k8s schedule
    parser_schedule = subparsers.add_parser(
        "schedule",
        help="Manage schedules",
        parents=[global_parser]
    )
    parser_schedule.set_defaults(func=_help_func(parser=parser_schedule))
    parser_schedule_subparsers = parser_schedule.add_subparsers()

    # maro k8s schedule start
    from maro.cli.k8s.schedule import start_schedule
    parser_schedule_start = parser_schedule_subparsers.add_parser(
        "start",
        help="Start a schedule",
        examples=CliExamples.MARO_K8S_SCHEDULE_START,
        parents=[global_parser]
    )
    parser_schedule_start.add_argument("cluster_name", help="Name of the cluster")
    parser_schedule_start.add_argument("deployment_path", help="Path of the schedule deployment")
    parser_schedule_start.set_defaults(func=start_schedule)

    # maro k8s schedule stop
    from maro.cli.k8s.schedule import stop_schedule
    parser_schedule_stop = parser_schedule_subparsers.add_parser(
        "stop",
        help="Stop a schedule",
        examples=CliExamples.MARO_K8S_SCHEDULE_STOP,
        parents=[global_parser]
    )
    parser_schedule_stop.add_argument("cluster_name", help="Name of the cluster")
    parser_schedule_stop.add_argument("schedule_name", help="Name of the schedule")
    parser_schedule_stop.set_defaults(func=stop_schedule)

    # maro k8s status
    from maro.cli.k8s.status import status
    parser_status = subparsers.add_parser(
        "status",
        help="Get status of the cluster",
        examples=CliExamples.MARO_K8S_STATUS,
        parents=[global_parser]
    )
    parser_status.add_argument("cluster_name", help="Name of the cluster")
    parser_status.set_defaults(func=status)

    # maro k8s template
    from maro.cli.k8s.template import template
    parser_template = subparsers.add_parser(
        "template",
        help="Get deployment templates",
        examples=CliExamples.MARO_K8S_TEMPLATE,
        parents=[global_parser]
    )
    parser_template.add_argument("export_path", default="./", help="Path of the export directory")
    parser_template.set_defaults(func=template)


def load_parser_data(prev_parser: ArgumentParser, global_parser: ArgumentParser):
    data_cmd_sub_parsers = prev_parser.add_subparsers()

    # BUILD
    from maro.cli.data_pipeline.utils import convert
    build_cmd_parser = data_cmd_sub_parsers.add_parser(
        "build",
        fromfile_prefix_chars="@",
        help="Build csv file to a strong type tight binary file.",
        parents=[global_parser])

    build_cmd_parser.add_argument(
        "--meta",
        type=str,
        required=True,
        help="Metafile for binary file building.")

    build_cmd_parser.add_argument(
        "--file",
        type=str,
        required=True,
        nargs="+",
        help="""
        Path to original csv file(s) used to build,
        you can save your files' name into a file and call with prefix @ to read files list from your file,
        like 'maro data build --meta meta.yml --output o.bin --file @files.txt'
        or just convert 1 file like 'maro data build --meta meta.yml --output o.bin --file input_file.csv'
        """)

    build_cmd_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path (with file name) to dump the binary file.")

    build_cmd_parser.add_argument(
        "--start-timestamp",
        dest="start_timestamp",
        type=int,
        default=None,
        required=False,
        help="""
        Specified start timestamp (in UTC) for binary file,
        then this timestamp will be considered as tick=0 for binary reader,
        this can be used to adjust the reader pipeline.
        """)

    build_cmd_parser.set_defaults(func=convert)


def load_parser_meta(prev_parser: ArgumentParser, global_parser: ArgumentParser):
    meta_cmd_sub_parsers = prev_parser.add_subparsers()
    # Deploy
    from maro.cli.data_pipeline.data_process import meta_deploy
    deploy_cmd_parser = meta_cmd_sub_parsers.add_parser(
        "deploy",
        help="Deploy data files for MARO.",
        parents=[global_parser])

    deploy_cmd_parser.set_defaults(func=meta_deploy)


def load_parser_inspector(prev_parser: ArgumentParser, global_parser: ArgumentParser):
    inspector_cmd_sub_parsers = prev_parser.add_subparsers()

    from maro.cli.inspector.env_data_process import start_vis
    dashboard_cmd_parser = inspector_cmd_sub_parsers.add_parser(
        "dashboard",
        fromfile_prefix_chars="@",
        help="Dashboard of selected env displayed.",
        parents=[global_parser]
    )

    dashboard_cmd_parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="Folder path to load data, should be root path of snapshot folders. e.g. ~/project_root/dump_files/"
    )

    dashboard_cmd_parser.add_argument(
        "--force",
        type=str,
        required=False,
        default="True",
        help="Overwrite the generated summary data or not: True/False."
    )

    dashboard_cmd_parser.set_defaults(func=start_vis)

    from maro.cli.maro_real_time_vis.start_maro_geo_vis import start_geo_vis
    geo_cmd_parser = inspector_cmd_sub_parsers.add_parser(
        "geo",
        fromfile_prefix_chars="@",
        help="Geographic data display.",
        parents=[global_parser]
    )

    geo_cmd_parser.add_argument(
        "--start",
        type=str,
        help="Kind of container expected to start, Database or Service.",
        required=True
    )

    geo_cmd_parser.add_argument(
        "--experiment_name",
        type=str,
        required=False,
        help="Name of experiment expected to be displayed."
    )

    geo_cmd_parser.add_argument(
        "--front_end_port",
        type=int,
        required=False,
        help="Specified port of front_end."
    )

    geo_cmd_parser.set_defaults(func=start_geo_vis)


def load_parser_project(prev_parser: ArgumentParser, global_parser: ArgumentParser):
    sub_parsers = prev_parser.add_subparsers()

    from maro.cli.project_generator.project_generator import new_project

    # maro project new
    new_cmd_parser = sub_parsers.add_parser(
        "new",
        help="Generate a new project under current directory to work with MARO.",
        parents=[global_parser]
    )

    # This command do not accept arguments as normal, instead with a simple wizard.
    new_cmd_parser.set_defaults(func=new_project)


def load_parser_admin(prev_parser: ArgumentParser, global_parser: ArgumentParser):
    sub_parsers = prev_parser.add_subparsers()
    # Start
    from maro.cli.utils.node_admin import start_admin
    start_parser = sub_parsers.add_parser(
        "start",
        help="Start MARO admin web server.",
        parents=[global_parser])

    start_parser.set_defaults(func=start_admin)

    # Stop
    from maro.cli.utils.node_admin import stop_admin
    stop_parser = sub_parsers.add_parser(
        "stop",
        help="Stop MARO admin web server.",
        parents=[global_parser])

    stop_parser.set_defaults(func=stop_admin)

    # Requirements
    from maro.cli.utils.node_admin import requirements_admin
    req_parser = sub_parsers.add_parser(
        "req",
        help="Install requirements for MARO admin web server.",
        parents=[global_parser])

    req_parser.set_defaults(func=requirements_admin)


def _help_func(parser):
    def wrapper(*args, **kwargs):
        parser.print_help()

    return wrapper


def _get_actual_args(namespace: Namespace) -> dict:
    actual_args = vars(deepcopy(namespace))
    return actual_args


if __name__ == '__main__':
    main()
