# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from multiprocessing import Manager, Process, Queue
from os import getcwd
from typing import Dict, List

from maro.communication import Proxy, SessionMessage
from maro.rl.utils import MsgKey, MsgTag
from maro.utils import Logger

# default group name for the cluster consisting of a policy manager and all policy hosts.
# If data parallelism is enabled, the gradient workers will also belong in this group.
DEFAULT_POLICY_GROUP = "policy_group_default"


class TaskQueueClient(object):
    """Task queue client for policies to interact with task queue."""
    def __init__(self):
        # TODO
        self._proxy = None

    def set_proxy(self, proxy):
        self._proxy = proxy

    def create_proxy(self, *args, **kwargs):
        self._proxy = Proxy(*args, **kwargs)

    def request_workers(self, task_queue_server_name="TASK_QUEUE"):
        """Request remote gradient workers from task queue to perform data parallelism."""
        worker_req = self._proxy.send(SessionMessage(MsgTag.REQUEST_WORKER, self._proxy.name, task_queue_server_name))
        worker_list = worker_req[0].body[MsgKey.WORKER_ID_LIST]
        return worker_list

    # TODO: rename this method
    def submit(self, worker_id_list: List, batch_list: List, policy_state: Dict, policy_name: str):
        """Learn a batch of data on several grad workers."""
        msg_dict = defaultdict(lambda: defaultdict(dict))
        loss_info_by_policy = {policy_name: []}
        for worker_id, batch in zip(worker_id_list, batch_list):
            msg_dict[worker_id][MsgKey.GRAD_TASK][policy_name] = batch
            msg_dict[worker_id][MsgKey.POLICY_STATE][policy_name] = policy_state
            # data-parallel by multiple remote gradient workers
            self._proxy.isend(SessionMessage(
                MsgTag.COMPUTE_GRAD, self._proxy.name, worker_id, body=msg_dict[worker_id]))
        dones = 0
        for msg in self._proxy.receive():
            if msg.tag == MsgTag.COMPUTE_GRAD_DONE:
                for policy_name, loss_info in msg.body[MsgKey.LOSS_INFO].items():
                    if isinstance(loss_info, list):
                        loss_info_by_policy[policy_name] += loss_info
                    elif isinstance(loss_info, dict):
                        loss_info_by_policy[policy_name].append(loss_info)
                    else:
                        raise TypeError(f"Wrong type of loss_info: {type(loss_info)}")
                dones += 1
                if dones == len(msg_dict):
                    break
        return loss_info_by_policy

    def exit(self):
        if hasattr(self, '_proxy'):
            self._proxy.close()


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
    peers = {"policy_host": num_hosts, "grad_worker": num_workers, "policy_manager": 1}
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
