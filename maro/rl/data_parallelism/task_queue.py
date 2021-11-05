# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from multiprocessing import Manager, Process, Queue, managers
from typing import Dict, List

from maro.communication import Proxy, SessionMessage
from maro.rl.utils import MsgKey, MsgTag
from maro.utils import DummyLogger, Logger

# default group name for the cluster consisting of a policy manager and all policy hosts.
# If data parallelism is enabled, the gradient workers will also belong in this group.
DEFAULT_POLICY_GROUP = "policy_group_default"


class TaskQueueClient(object):
    """Task queue client for policies to interact with task queue."""
    def __init__(self):
        # TODO: singleton
        self._proxy = None

    def set_proxy(self, proxy):
        self._proxy = proxy

    def create_proxy(self, *args, **kwargs):
        self._proxy = Proxy(*args, **kwargs)

    def request_workers(self, task_queue_server_name="TASK_QUEUE"):
        """Request remote gradient workers from task queue to perform data parallelism."""
        worker_req = self._proxy.send(
            SessionMessage(MsgTag.REQUEST_WORKER, self._proxy.name, task_queue_server_name))[0]
        worker_list = worker_req.body[MsgKey.WORKER_ID_LIST]
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
    single_task_limit: float = 0.5,
    group: str = DEFAULT_POLICY_GROUP,
    proxy_kwargs: dict = {},
    logger: Logger = DummyLogger()
):
    num_workers = len(worker_ids)
    if num_hosts == 0:
        # for multi-process mode
        num_hosts = num_policies

    # Proxy
    peers = {"policy_host": num_hosts, "grad_worker": num_workers, "policy_manager": 1}
    proxy = Proxy(group, "task_queue", peers, component_name="TASK_QUEUE", **proxy_kwargs)

    assert single_task_limit > 0.0 and single_task_limit <= 1.0, "single_task_limit" \
        f"should be greater than 0.0 and less than 1.0, but {single_task_limit} instead."
    MAX_WORKER_SINGLE_TASK = max(1, int(single_task_limit * num_workers))

    def consume(task_pending: managers.ListProxy, task_assigned: Queue, worker_available_status: Dict, signal: Dict):
        recent_used_workers = []
        while not signal["EXIT"]:
            if len(task_pending) == 0:
                continue

            # limit the worker number of a single task.
            max_worker_num = min(1 + num_workers // len(task_pending), MAX_WORKER_SINGLE_TASK)

            assigned_workers = []
            # select from recent used workers first
            worker_candidates = [worker_id for worker_id in recent_used_workers if worker_id in worker_available_status]
            worker_candidates += list(set(worker_available_status.keys()) - set(recent_used_workers))
            for worker_id in worker_candidates:
                if worker_available_status[worker_id]:
                    assigned_workers.append(worker_id)
                    worker_available_status[worker_id] = False
                    # update recent used workers
                    if worker_id in recent_used_workers:
                        recent_used_workers.remove(worker_id)
                    recent_used_workers.insert(0, worker_id)
                if len(assigned_workers) >= max_worker_num:
                    break

            if not assigned_workers:
                continue

            task_pending.sort(key=get_priority, reverse=True)  # sort in descending priority
            msg, priority = task_pending.pop(0)
            task_assigned.put({"msg": msg, "worker_id_list": assigned_workers})

    # Process
    manager = Manager()
    signal = manager.dict()
    worker_available_status = manager.dict()  # workers are available or not.
    task_pending = manager.list()
    task_assigned = Queue()

    for worker_id in worker_ids:
        worker_available_status[worker_id] = True

    signal["EXIT"] = False

    cons = Process(target=consume, args=(task_pending, task_assigned, worker_available_status, signal))
    cons.start()

    while not signal["EXIT"]:
        # receive message with time limit 10ms
        for msg in proxy.receive(timeout=10):
            if msg.tag == MsgTag.EXIT:
                logger.info("Exiting...")
                proxy.close()
                signal["EXIT"] = True
                break
            elif msg.tag == MsgTag.REQUEST_WORKER:
                priority = 1.0
                task_pending.append((msg, priority))
            elif msg.tag == MsgTag.RELEASE_WORKER:
                worker_id = msg.body[MsgKey.WORKER_ID]
                worker_available_status[worker_id] = True
            # TODO: support add/remove workers
            elif msg.tag == MsgTag.ADD_WORKER:
                worker_id = msg.body[MsgKey.WORKER_ID]
                worker_available_status[worker_id] = True
                num_workers += 1
            elif msg.tag == MsgTag.REMOVE_WORKER:
                worker_id = msg.body[MsgKey.WORKER_ID]
                if worker_id in worker_available_status:
                    worker_available_status.pop(worker_id)
                    num_workers -= 1
            else:
                raise TypeError

        # dequeue finished tasks
        while task_assigned.qsize() > 0:
            task = task_assigned.get()
            _msg, worker_id_list = task["msg"], task["worker_id_list"]
            msg_body = {MsgKey.WORKER_ID_LIST: worker_id_list}
            proxy.reply(_msg, tag=MsgTag.ASSIGN_WORKER, body=msg_body)

    cons.join()


# magic methods like lambda is not supported in multiprocessing
def get_priority(x):
    return x[1]
