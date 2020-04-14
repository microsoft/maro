import sys 
import os
sys.path.append("..") 
from utils import auto_sync

# euid = os.geteuid()
# if euid != 0:
#     print("Script not started as root. Running sudo..")
#     args = ['sudo', sys.executable] + sys.argv + [os.environ]
#     # the next line replaces the currently-running process with the sudo
#     os.execlpe('sudo', *args)
# print(os.geteuid())

if __name__ == "__main__":
   # src = os.environ['PYTHONPATH']
    src = 'home/tianyi/maro/'
    dest = '/codepoint/'
    auto_sync(src, dest)
