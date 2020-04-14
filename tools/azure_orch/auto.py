import sys
import os
sys.path.append("..") 
from utils import auto_sync

if __name__ == "__main__":
    src = "/home/tianyi/maro/"
    dest = "/codepoint/"
    auto_sync(src, dest)