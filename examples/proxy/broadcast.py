# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import multiprocessing as mp

from maro.communication import Proxy, SessionType


def worker(group_name):
    """
    The main worker logic includes initialize proxy and handle jobs from the master.

    Args:
        group_name (str): Identifier for the group of all communication components.
    """
    proxy = Proxy(
        group_name=group_name,
        component_type="worker",
        expected_peers={"master": 1},
    )
    counter = 0
    print(f"{proxy.name}'s counter is {counter}.")

    # Nonrecurring receive the message from the proxy.
    msg = proxy.receive_once()
    print(f"{proxy.name} received message from {msg.source}.")

    if msg.tag == "INC":
        counter += 1
        print(f"{proxy.name} receive INC request, {proxy.name}'s count is {counter}.")
        proxy.reply(message=msg, tag="done")


def master(group_name: str, worker_num: int, is_immediate: bool = False):
    """
    The main master logic includes initialize proxy and allocate jobs to workers.

    Args:
        group_name (str): Identifier for the group of all communication components,
        worker_num (int): The number of workers,
        is_immediate (bool): If True, it will be an async mode; otherwise, it will be an sync mode.
            Async Mode: The proxy only returns the session id for sending messages. Based on the local task priority,
                        you can do something with high priority before receiving replied messages from peers.
            Sync Mode: It will block until the proxy returns all the replied messages.
    """
    proxy = Proxy(
        group_name=group_name,
        component_type="master",
        expected_peers={"worker": worker_num},
    )

    if is_immediate:
        session_ids = proxy.ibroadcast(
            component_type="worker",
            tag="INC",
            session_type=SessionType.NOTIFICATION,
        )
        # Do some tasks with higher priority here.
        replied_msgs = proxy.receive_by_id(session_ids, timeout=-1)
    else:
        replied_msgs = proxy.broadcast(
            component_type="worker",
            tag="INC",
            session_type=SessionType.NOTIFICATION,
            timeout=-1,
        )

    for msg in replied_msgs:
        print(
            f"{proxy.name} get receive notification from {msg.source} with "
            f"message session stage {msg.session_stage}.",
        )


if __name__ == "__main__":
    """
    This is a single-host multiprocess program used to simulate the communication in the distributed system.
    For the completed usage experience of the distributed cluster, please use the MARO CLI.
    """
    mp.set_start_method("spawn")

    group_name = "proxy_broadcast_INC_example"
    worker_number = 5
    is_immediate = True

    workers = mp.Pool(worker_number)

    master_process = mp.Process(target=master, args=(group_name, worker_number, is_immediate))
    master_process.start()

    workers.map(worker, [group_name] * worker_number)
    workers.close()

    master_process.join()
    workers.join()
