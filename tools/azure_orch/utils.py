import time
import logging
import os

from dirsync import sync

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s - %(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

class CodeFileEventHandler(FileSystemEventHandler):
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

class LogFileEventHandler(FileSystemEventHandler):
    def __init__(self, src, dest):
        FileSystemEventHandler.__init__(self)
        self._src = src
        self._dest = dest

    def on_created(self, event):
        sync(self._src, self._dest, 'sync')
        logging.info("[create] {0} sync to: {1}".format(event.src_path, self._dest))
    
    def on_modified(self, event):
        sync(self._src, self._dest, 'sync')
        logging.info("[modify] {0} sync to: {1}".format(event.src_path, self._dest))


def auto_sync(src, dest):
    sync(src, dest, 'sync', purge=True)
    code_observer = Observer()
    log_observer = Observer()
    code_event_handler = CodeFileEventHandler(src, dest)
    log_event_handler = LogFileEventHandler(dest, src)
    code_observer.schedule(code_event_handler, src, True)
    log_observer.schedule(log_event_handler, dest, True)
    code_observer.start()
    log_observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        code_observer.stop()
        log_observer.stop()

    code_observer.join()
    log_observer.join()


