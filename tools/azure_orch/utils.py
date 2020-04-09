import time
import logging
import os

from dirsync import sync

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
from psutil import cpu_percent, cpu_count, virtual_memory
from socket import gethostname
import redis

import json

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s - %(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

class FileEventHandler(FileSystemEventHandler):
    def __init__(self, src, dest):
        FileSystemEventHandler.__init__(self)
        self._src = src
        self._dest = dest
        
    def on_moved(self, event):
        sync(self._src, self._dest, 'sync', purge=True)
        logging.info("[move] from {0} to {1} sync to: {3}".format(event.src_path, event.dest_path, self._dest))

    def on_created(self, event):
        sync(self._src, self._dest, 'sync', purge=True)
        logging.info("[create] {0} sync to: {1}".format(event.src_path, self._dest))


    def on_deleted(self, event):
        sync(self._src, self._dest, 'sync', purge=True)
        logging.info("[delete] {0} sync to: {1}".format(event.src_path, self._dest))


    def on_modified(self, event):
        sync(self._src, self._dest, 'sync', purge=True)
        logging.info("[modify] {0} sync to: {1}".format(event.src_path, self._dest))

def auto_sync(src, dest):
    sync(src, dest, 'sync', purge=True)
    observer = Observer()
    event_handler = FileEventHandler(src, dest)
    observer.schedule(event_handler, src, True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


class Prob():
    def __init__(self):
        self._redis_connection = redis.StrictRedis(host=os.environ['redis_address'], port=os.environ['redis_port'])
    
    def prob(self):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        meminfo = nvmlDeviceGetMemoryInfo(handle)
        free_GPU_mem = meminfo.free / (1024 * 1024)

        mem = virtual_memory()
        free_mem = (mem.total - mem.used) / (1024 * 1024)

        free_CPU_cores = (100 - cpu_percent()) / 100 * cpu_count()

        free_resources = {"free_GPU_mem" : free_GPU_mem,
                          "free_mem": free_mem,
                          "free_CPU_cores": free_CPU_cores}

        self._redis_connection.hset("resources", gethostname(), json.dumps(free_resources))