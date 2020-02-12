import argparse
import os

def _env_func(args):
    print('envs', args.envs_action)

def _dashboard_func(args):
    option_exists = False
    if args.unzip:
        print('args.unzip')
        option_exists = True
    if args.start:
        print('args.start')
        option_exists = True
    if args.stop:
        print('args.stop')
        option_exists = True
    if args.build:
        print('args.build')
        option_exists = True
    if not option_exists:
        parser_dashboard.print_help()

    # if args.dashboard_action == 'unzip':
    #     print('unzip',args.dashboard_action)
    # elif args.dashboard_action == 'start':
    #     print('start',args.dashboard_action)
    # elif args.dashboard_action == 'stop':
    #     print('stop',args.dashboard_action)
    # # elif args.dashboard_action == 'help':
    # #     print('default or 'unzip' to extract dashboard resources to current folder. 'start' to start dashboard service. 'stop' to stop dashboard service. 'build' to build docker for dashboard service.')
    # elif args.dashboard_action == 'build':
    #     print('build',args.dashboard_action)
    # else:
    #     parser_dashboard.print_help()
def _help_func(args):
    parser.print_help()
    print(f'{parser} not found')


parser = argparse.ArgumentParser('maro', description ='MARO cli interface', add_help = False)
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
parser_dashboard.add_argument('-u', '--unzip', action='store_true',
                            help='extract to current directory')
parser_dashboard.add_argument('-s', '--start', action='store_true',
                            help='start to current directory')
parser_dashboard.add_argument('-t', '--stop', action='store_true',
                            help='stop to current directory')
parser_dashboard.add_argument('-b', '--build', action='store_true',
                            help='build to current directory')

# parser_dashboard.add_argument('dashboard_action', nargs='?', choices=['unzip', 'start', 'stop', 'help', 'build'], default='help', const='unzip', metavar='ACTION',
#                                 help='default or 'unzip' to extract dashboard resources to current folder. 'start' to start dashboard service. 'stop' to stop dashboard service. 'build' to build docker for dashboard service.')
parser_dashboard.set_defaults(func=_dashboard_func)

args = parser.parse_args()
args.func(args)
