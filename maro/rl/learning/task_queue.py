# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from multiprocessing import Manager, Process, Queue
from os import getcwd
from typing import Dict, List

from maro.communication import Proxy
from maro.rl.utils import MsgKey, MsgTag
from maro.utils import Logger

# default group name for the cluster consisting of a policy manager and all policy hosts.
# If data parallelism is enabled, the gradient workers will also belong in this group.
DEFAULT_POLICY_GROUP = "policy_group_default"


def task_queue(
    worker_ids: List[str],
    num_hosts: int,
    num_policies: int,
    group: str = DEFAULT_POLICY_GROUP,
    proxy_kwargs: dict = {},
    log_dir: str = getcwd()
):
    num_workers = len(worker_ids)
    if num_hosts == 0:
        # for multi-process mode
        num_hosts = num_policies

    # Proxy
    peers = {"policy_host": num_hosts, "grad_worker": num_workers}
    proxy = Proxy(group, "task_queue", peers, component_name="TASK_QUEUE", **proxy_kwargs)
    logger = Logger(proxy.name, dump_folder=log_dir)

    def consume(task_waiting: Queue, task_done: Queue, worker_status: Dict, signal: Dict):
        while not signal["EXIT"]:
            if task_waiting.qsize() == 0:
                continue

            # allow 50% workers to a single task at most.
            max_worker_num = 1 + num_workers // max(2, task_waiting.qsize())

            worker_id_list = []
            for worker_id, idle in worker_status.items():
                if idle:
                    worker_id_list.append(worker_id)
                    worker_status[worker_id] = False
                if len(worker_id_list) >= max_worker_num:
                    break

            if len(worker_id_list) == 0:
                continue

            num_idle = 0
            for worker_id in dict(worker_status):
                if worker_status[worker_id]:
                    num_idle += 1

            msg = task_waiting.get()
            task_done.put({"msg": msg, "worker_id_list": worker_id_list})

    # Process
    manager = Manager()
    signal = manager.dict()
    worker_status = manager.dict()
    task_waiting = Queue()
    task_done = Queue()

    for worker_id in worker_ids:
        worker_status[worker_id] = True

    signal["EXIT"] = False

    c = Process(target=consume, args=(task_waiting, task_done, worker_status, signal))
    c.start()

    while not signal["EXIT"]:
        # receive message with time limit 10ms
        for msg in proxy.receive(timeout=10):
            if msg.tag == MsgTag.EXIT:
                logger.info("Exiting...")
                proxy.close()
                signal["EXIT"] = True
                break
            elif msg.tag == MsgTag.REQUEST_WORKER:
                task_waiting.put(msg)
            elif msg.tag == MsgTag.RELEASE_WORKER:
                worker_id = msg.body[MsgKey.WORKER_ID]
                worker_status[worker_id] = True
            # TODO: support add/remove workers
            elif msg.tag == MsgTag.ADD_WORKER:
                worker_id = msg.body[MsgKey.WORKER_ID]
                worker_status[worker_id] = True
                num_workers += 1
            elif msg.tag == MsgTag.REMOVE_WORKER:
                worker_id = msg.body[MsgKey.WORKER_ID]
                if worker_id in worker_status:
                    worker_status.pop(worker_id)
                    num_workers -= 1
            else:
                raise TypeError

        # dequeue finished tasks
        while task_done.qsize() > 0:
            task = task_done.get()
            _msg, worker_id_list = task["msg"], task["worker_id_list"]
            msg_body = {MsgKey.WORKER_ID_LIST: worker_id_list}
            proxy.reply(_msg, tag=MsgTag.ASSIGN_WORKER, body=msg_body)

    c.join()
