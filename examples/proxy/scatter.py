# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import multiprocessing as mp

import numpy as np

from maro.communication import Proxy, SessionType


def summation_worker(group_name):
    """
    The main worker logic includes initialize proxy and handle sum jobs from the master.

    Args:
        group_name (str): Identifier for the group of all communication components.
    """
    proxy = Proxy(
        group_name=group_name,
        component_type="sum_worker",
        expected_peers={"master": 1},
    )

    # Nonrecurring receive the message from the proxy.
    msg = proxy.receive_once()
    print(f"{proxy.name} received message from {msg.source}. the payload is {msg.body}.")

    if msg.tag == "job":
        replied_payload = sum(msg.body)
        proxy.reply(message=msg, tag="sum", body=replied_payload)


def multiplication_worker(group_name):
    """
    The main worker logic includes initialize proxy and handle multiply jobs from the master.

    Args:
        group_name (str): Identifier for the group of all communication components.
    """
    proxy = Proxy(
        group_name=group_name,
        component_type="multiply_worker",
        expected_peers={"master": 1},
    )

    # Nonrecurring receive the message from the proxy.
    msg = proxy.receive_once()
    print(f"{proxy.name} receive message from {msg.source}. the payload is {msg.body}.")

    if msg.tag == "job":
        replied_payload = np.prod(msg.body)
        proxy.reply(message=msg, tag="multiply", body=replied_payload)


def master(group_name: str, sum_worker_number: int, multiply_worker_number: int, is_immediate: bool = False):
    """
    The main master logic includes initialize proxy and allocate jobs to workers.

    Args:
        group_name (str): Identifier for the group of all communication components,
        sum_worker_number (int): The number of sum workers,
        multiply_worker_number (int): The number of multiply workers,
        is_immediate (bool): If True, it will be an async mode; otherwise, it will be an sync mode.
            Async Mode: The proxy only returns the session id for sending messages. Based on the local task priority,
                        you can do something with high priority before receiving replied messages from peers.
            Sync Mode: It will block until the proxy returns all the replied messages.
    """
    proxy = Proxy(
        group_name=group_name,
        component_type="master",
        expected_peers={
            "sum_worker": sum_worker_number,
            "multiply_worker": multiply_worker_number,
        },
    )

    sum_list = np.random.randint(0, 10, 100)
    multiple_list = np.random.randint(1, 10, 20)
    print("Generate random sum/multiple list with length 100.")

    # Assign sum tasks for summation workers.
    destination_payload_list = []
    for idx, peer in enumerate(proxy.peers["sum_worker"]):
        data_length_per_peer = int(len(sum_list) / len(proxy.peers["sum_worker"]))
        destination_payload_list.append((peer, sum_list[idx * data_length_per_peer : (idx + 1) * data_length_per_peer]))

    # Assign multiply tasks for multiplication workers.
    for idx, peer in enumerate(proxy.peers["multiply_worker"]):
        data_length_per_peer = int(len(multiple_list) / len(proxy.peers["multiply_worker"]))
        destination_payload_list.append(
            (peer, multiple_list[idx * data_length_per_peer : (idx + 1) * data_length_per_peer]),
        )

    if is_immediate:
        session_ids = proxy.iscatter(
            tag="job",
            session_type=SessionType.TASK,
            destination_payload_list=destination_payload_list,
        )
        # Do some tasks with higher priority here.
        replied_msgs = proxy.receive_by_id(session_ids, timeout=-1)
    else:
        replied_msgs = proxy.scatter(
            tag="job",
            session_type=SessionType.TASK,
            destination_payload_list=destination_payload_list,
            timeout=-1,
        )

    sum_result, multiply_result = 0, 1
    for msg in replied_msgs:
        if msg.tag == "sum":
            print(f"{proxy.name} receive message from {msg.source} with the sum result {msg.body}.")
            sum_result += msg.body
        elif msg.tag == "multiply":
            print(f"{proxy.name} receive message from {msg.source} with the multiply result {msg.body}.")
            multiply_result *= msg.body

    # Check task result correction.
    assert sum(sum_list) == sum_result
    assert np.prod(multiple_list) == multiply_result


if __name__ == "__main__":
    """
    This is a single-host multiprocess program used to simulate the communication in the distributed system.
    For the completed usage experience of the distributed cluster, please use the MARO CLI.
    """
    mp.set_start_method("spawn")

    group_name = "proxy_scatter_mixed_worker_test"
    sum_worker_number = 5
    multiply_worker_number = 5
    is_immediate = False

    # Worker's pool for sum_worker and prod_worker.
    workers = mp.Pool(sum_worker_number + multiply_worker_number)

    master_process = mp.Process(
        target=master,
        args=(group_name, sum_worker_number, multiply_worker_number, is_immediate),
    )
    master_process.start()

    for s in range(sum_worker_number):
        workers.apply_async(func=summation_worker, args=(group_name,))

    for m in range(multiply_worker_number):
        workers.apply_async(func=multiplication_worker, args=(group_name,))

    workers.close()

    master_process.join()
    workers.join()
