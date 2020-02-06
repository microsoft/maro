import argparse

parser = argparse.ArgumentParser("maro cli interface")
parser.add_argument("-envs", action="store_true",
                    help="Show available environment settings")
parser.add_argument("-dashboard", nargs='?', choices=['unzip', 'start', 'stop', 'no_action', 'build'], default='no_action', const='unzip', metavar='ACTION',
                    help="default or 'unzip' to extract dashboard resources to current folder. 'start' to start dashboard service. 'stop' to stop dashboard service. 'build' to build docker for dashboard service.")

args = parser.parse_args()

if args.envs:
    print('print_envs', args.envs)
if args.dashboard == 'unzip':
    print('unzip',args.dashboard)
elif args.dashboard == 'start':
    print('start',args.dashboard)
elif args.dashboard == 'stop':
    print('stop',args.dashboard)
elif args.dashboard == 'no_action':
    pass
elif args.dashboard == 'build':
    print('build',args.dashboard)
else:
    print("default or 'unzip' to extract dashboard resources to current folder. 'start' to start dashboard service. 'stop' to stop dashboard service. 'build' to build docker for dashboard service.")
