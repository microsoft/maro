# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


# NOTE
import sys
import argparse
import site
import tarfile
import os
import io
import platform
import yaml
import subprocess
import socket
import webbrowser
from requests import get
from maro.simulator.utils.common import get_available_envs

from tools.azure_orch.scripts.provision import create_resource_group, increase_resource_group, decrease_resource_group, stop_workers, start_workers
from tools.azure_orch.scripts.docker import launch_job
from tools.azure_orch.scripts.utils import sync_code, pull_log, generate_job_config, sync_resource_group_info


# static variables for calling from subfunctions
parser = None
parser_dashboard = None
parser_dist = None

def print_envs():
    '''
    Show available env configurations in package

    Args:
        None

    Returns:
        None
    '''
    envs = get_available_envs()

    for env in envs:
        print(f'scenario: {env["scenario"]}, topology: {env["topology"]}')


def main():
    '''
    Set in setup.py

    .. codeblock:: python

    entry_points={
        "console_scripts": [
            \'maro=maro.cli.maro:main\',
        ]
    }

    Parse parameters like:

    1. maro dashboard -> extact dashboard resource files to current directory, or start, stop the dashboard for MARO

    2. maro envs -> show available env configuration

    3. maro [-h/--help] -> show help for maro cli

    Args:
        None

    Returns:
        None
    '''
    global parser
    global parser_dashboard
    global parser_dist

    parser = argparse.ArgumentParser('maro', description='MARO cli interface', add_help = False)
    parser.add_argument('-h','--help', action='store_true',
        help='Show help string for MARO cli interface')
    parser.set_defaults(func=_help_func)

    subparsers = parser.add_subparsers(metavar='CLI_MODULE')
    parser_envs = subparsers.add_parser(
        'envs', help='Show available environment settings')
    parser_envs.add_argument('envs_action', action='store_true',
                             help='Show available environment settings')
    parser_envs.set_defaults(func=_env_func)

    parser_dashboard = subparsers.add_parser(
        'dashboard', help='Extract dashboard file and start or stop dashboard for MARO')
    parser_dashboard.add_argument('-e', '--extract', action='store_true',
                                help='Extract dashboard files to current directory')
    parser_dashboard.add_argument('-s', '--start', action='store_true',
                                help='Start dashboard for MARO, if dashboard files are extracted to current directory')
    parser_dashboard.add_argument('-t', '--stop', action='store_true',
                                help='Stop dashboard for MARO, if dashboard files are extracted to current directory')
    parser_dashboard.add_argument('-b', '--build', action='store_true',
                                help='Rebuild docker image for dashboard, if dashboard files are extracted to current directory')
    parser_dashboard.set_defaults(func=_dashboard_func)

    parser_dist = subparsers.add_parser(
        'dist', help="create vm and launch jobs for MARO")
    parser_dist.add_argument('-crg', '--create_resource_group', action='store_true',
                                help='create a new resource group')
    parser_dist.add_argument('-irg', '--increase_resource_group', action='store_true',
                                help='create extra workers for a resource group')
    parser_dist.add_argument('-drg', '--decrease_resource_group', action='store_true',
                                help='eliminate extra workers for a resource group')
    parser_dist.add_argument('-startw', '--start_workers', action='store_true',
                                help='start all workers in a resource group')
    parser_dist.add_argument('-stopw', '--stop_workers', action='store_true',
                                help='stop all workers in a resource group')
    parser_dist.add_argument('-sc', '--sync_code', action='store_true',
                                help='sync your current code to god if you are in dev machine, create a codebase if not exist')
    parser_dist.add_argument('-pl', '--pull_log', action='store_true',
                                help='pull log data from the god if you are in dev machine')  
    parser_dist.add_argument('-g', '--generate_job_config', action='store_true',
                                help='generate job configs to launch jobs')
    parser_dist.add_argument('-l', '--launch_job', action='store_true',
                                help='launch jobs')
    parser_dist.add_argument('-srg', '--sync_resource_group_info', action='store_true',
                                help='get the resource group info from the god machine to use that resource group if you are not lucy')
    parser_dist.set_defaults(func=_dist_func)

    args = parser.parse_args()
    args.func(args)


def _help_func(args):
    parser.print_help()


def _env_func(args):
    print_envs()


def _dashboard_func(args):
    option_exists = False
    if args.unzip:
        print('Unzip dashboard files')
        ext_dashboard()
        option_exists = True

    if args.start:
        print('Start dashboard')
        start_dashboard()
        option_exists = True

    if args.stop:
        print('Stop dashboard')
        stop_dashboard()
        option_exists = True

    if args.build:
        print('Rebuild docker image for dashboard')
        build_dashboard()
        option_exists = True

    if not option_exists:
        parser_dashboard.print_help()

def _dist_func(args):
    option_exists = False
    if args.create_resource_group:
        create_resource_group()
        option_exists = True
    elif args.increase_resource_group:
        increase_resource_group()
        option_exists = True
    elif args.decrease_resource_group:
        decrease_resource_group()
        option_exists = True
    elif args.start_workers:
        start_workers()
        option_exists = True
    elif args.stop_workers:
        stop_workers()
        option_exists = True
    elif args.generate_job_config:
        generate_job_config()
        option_exists = True
    elif args.launch_job:
        launch_job()
        option_exists = True
    elif args.sync_code:
        sync_code()
        option_exists = True
    elif args.pull_log:
        pull_log()
        option_exists = True
    elif args.sync_resource_group_info:
        sync_resource_group_info()
        option_exists = True

    if not option_exists:
        parser_dist.print_help()

def ext_dashboard():
    '''
        Extract dashboard server file to current directory.

        Args:
            None.

        Returns:
            None.
    '''
    print('Extracting maro dashboard data package.')
    data_path_g = os.path.join(sys.prefix, 'maro_dashboard', 'resource.tar.gz')
    data_path_user = os.path.join(
        site.USER_BASE, 'maro_dashboard', 'resource.tar.gz')
    if os.path.exists(data_path_g):
        my_data = tarfile.open(data_path_g, 'r:gz')
    elif os.path.exists(data_path_user):
        my_data = tarfile.open(data_path_user, 'r:gz')
    else:
        print('maro dashboard data package not found')
        sys.exit(0)
    my_data.extractall()
    my_data.close()


def start_dashboard():
    '''
        Start dashboard service 

        Args:
            None.

        Returns:
            None.
    '''

    print('Try to start dashboard service.')
    cwd = os.getcwd()
    all_files_exist = True
    for path in ['config', 'panels', 'provisioning', 'templates', 'docker-compose.yml', 'Dockerfile']:
        tar_path = os.path.join(cwd, path)
        if not os.path.exists(tar_path):
            print(f'{tar_path} not found')
            all_files_exist = False
    if not all_files_exist:
        print(f'Dashboard files not found, aborting...')
        return
    if not platform.system() == 'Windows':
        os.system(
            'mkdir -p ./data/grafana; CURRENT_UID=$(id -u):$(id -g) docker-compose up -d')
    else:
        os.system(
            'powershell.exe -windowstyle hidden "docker-compose up -d"', shell=True, start_new_session=True)

    localhosts = _get_ip_list()

    dashboard_port = '50303'

    yml_path = os.path.join(cwd, 'docker-compose.yml')
    if os.path.exists(yml_path):
        with io.open(yml_path, 'r') as in_file:
            raw_config = yaml.safe_load(in_file)
            if raw_config.get('services'):
                if raw_config['services'].get('grafana'):
                    if not raw_config['services']['grafana'].get('ports') is None:
                        if len(raw_config['services']['grafana']['ports']) > 0:
                            dashboard_port_tmp = raw_config['services']['grafana']['ports'][0].split(
                                ':')
                            if len(dashboard_port_tmp) > 0:
                                dashboard_port = dashboard_port_tmp[0]

    for localhost in localhosts:
        print(f'Dashboard Link:  http://{localhost}:{dashboard_port}')
        webbrowser.open(f'{localhost}:{dashboard_port}')


def stop_dashboard():
    '''
        Stop dashboard service 

        Args:
            None.

        Returns:
            None.
    '''

    print('Try to stop dashboard service.')
    cwd = os.getcwd()
    all_files_exist = True
    for path in ['docker-compose.yml']:
        tar_path = os.path.join(cwd, path)
        if not os.path.exists(tar_path):
            print(f'{tar_path} not found')
            all_files_exist = False
    if not all_files_exist:
        print(f'Dashboard files not found, aborting...')
        return
    if not platform.system() == 'Windows':
        os.system('docker-compose down')
    else:
        os.system('powershell.exe -windowstyle hidden "docker-compose down"')


def build_dashboard():
    '''
        Build docker for dashboard service 

        Args:
            None.

        Returns:
            None.
    '''

    print('Try to build docker for dashboard service.')
    cwd = os.getcwd()
    all_files_exist = True
    for path in ['config', 'panels', 'provisioning', 'templates', 'docker-compose.yml', 'Dockerfile']:
        tar_path = os.path.join(cwd, path)
        if not os.path.exists(tar_path):
            print(f'{tar_path} not found')
            all_files_exist = False
    if not all_files_exist:
        print(f'Dashboard files not found, aborting...')
        return
    if not platform.system() == 'Windows':
        os.system(
            'docker-compose build --no-cache')
    else:
        os.system(
            'powershell.exe -windowstyle hidden "docker-compose build --no-cache"', shell=True, start_new_session=True)


def _get_ip_list():
    print('Try to get ip list.')
    localhosts = []
    localhosts.append('localhost')

    try:
        ip = get('https://api.ipify.org').text
        if not ip is None:
            print('Public IP address:', ip)
            localhosts.append(ip)
    except Exception as e:
        print('Exception in getting public ip:', str(e))

    # REFERENCE https://www.chenyudong.com/archives/python-get-local-ip-graceful.html
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        if not ip in localhosts:
            print('Private IP address:', ip)
            localhosts.append(ip)
    except Exception as e:
        print('Exception in getting private ip:', str(e))
    finally:
        s.close()

    return localhosts


if __name__ == '__main__':
    main()
    