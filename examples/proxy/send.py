# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import multiprocessing as mp

import numpy as np

from maro.communication import Proxy, SessionMessage, SessionType


def worker(group_name):
    """
    The main worker logic includes initialize proxy and handle jobs from the master.

    Args:
        group_name (str): Identifier for the group of all communication components
    """
    proxy = Proxy(group_name=group_name,
                  component_type="worker",
                  expected_peers={"master": 1})

    # nonrecurring receive the message from the proxy.
    for msg in proxy.receive(is_continuous=False):
        print(f"{proxy.component_name} receive message from {msg.source}. the payload is {msg.payload}.")

        if msg.tag == "sum":
            replied_payload = sum(msg.payload)
            proxy.reply(received_message=msg, tag="sum", payload=replied_payload)


def master(group_name: str, is_immediate: bool = False):
    """
    The main master logic includes initialize proxy and allocate jobs to workers.

    Args:
        group_name (str): Identifier for the group of all communication components,
        is_immediate (bool): If True, it will be an async mode; otherwise, it will be an sync mode.
            Async Mode: The proxy only returns the session id for sending messages. Based on the local task priority,
                        you can do something with high priority before receiving replied messages from peers.
            Sync Mode: It will block until the proxy returns all the replied messages.
    """
    proxy = Proxy(group_name=group_name,
                  component_type="master",
                  expected_peers={"worker": 1})

    random_integer_list = np.random.randint(0, 100, 5)
    print(f"generate random integer list: {random_integer_list}.")

    for peer in proxy.peers["worker"]:
        message = SessionMessage(tag="sum",
                                 source=proxy.component_name,
                                 destination=peer,
                                 payload=random_integer_list,
                                 session_type=SessionType.TASK)
        if is_immediate:
            session_id = proxy.isend(message)
            # do some tasks with higher priority here.
            replied_msgs = proxy.receive_by_id(session_id)
        else:
            replied_msgs = proxy.send(message)

        for msg in replied_msgs:
            print(f"{proxy.component_name} receive {msg.source}, replied payload is {msg.payload}.")


if __name__ == "__main__":
    """
    This is a single-host multiprocess program used to simulate the communication in the distributed system.
    For the completed usage experience of the distributed cluster, please use the MARO CLI.
    """
    mp.set_start_method("spawn")

    group_name = "proxy_send_simple_example"
    is_immediate = False

    master_process = mp.Process(target=master, args=(group_name, is_immediate,))
    worker_process = mp.Process(target=worker, args=(group_name, ))
    master_process.start()
    worker_process.start()

    master_process.join()
    worker_process.join()
