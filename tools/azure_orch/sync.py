from dirsync import sync
import sys 
import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s - %(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

log = logging.getLogger('log')

euid = os.geteuid()
if euid != 0:
    print("Script not started as root. Running sudo..")
    args = ['sudo', sys.executable] + sys.argv + [os.environ]
    # the next line replaces the currently-running process with the sudo
    os.execlpe('sudo', *args)

if __name__ == "__main__":
    src = '/home/tianyi/maro/'
    dest = '/codepoint/'
    sync(src, dest, 'sync', purge=True, logger=log, exclude=(r".*log.*", ))
    sync(src, dest, 'sync', purge=True, logger=log, only=r".*log.*")
