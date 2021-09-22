# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from multiprocessing import Pipe, Process
from os import getcwd
from typing import Callable, Dict, List

from maro.communication import Proxy, SessionMessage, SessionType
from maro.rl.policy import RLPolicy, WorkerAllocator
from maro.rl.utils import MsgKey, MsgTag
from maro.utils import Logger


class AbsPolicyManager(ABC):
    """Facility that controls policy update and serves the latest policy states."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def update(self, rollout_info: Dict[str, list]):
        """Update policies using roll-out information.

        The roll-out information is grouped by policy name and may be either a training batch or a list of loss
        information dictionaries computed directly by roll-out workers.
        """
        raise NotImplementedError

    @abstractmethod
    def get_state(self):
        """Get the latest policy states."""
        raise NotImplementedError

    @abstractmethod
    def get_version(self):
        """Get the collective policy version."""
        raise NotImplementedError

    def server(self, group: str, num_actors: int, max_lag: int = 0, proxy_kwargs: dict = {}, log_dir: str = getcwd()):
        """Run a server process.

        The process serves the latest policy states to a set of remote actors and receives simulated experiences from
        them.

        Args:
            group (str): Group name for the cluster that includes the server and all actors.
            num_actors (int): Number of remote actors to collect simulation experiences.
            max_lag (int): Maximum policy version lag allowed for experiences collected from remote actors. Experiences
                collected using policy versions older than (current_version - max_lag) will be discarded. Defaults to 0,
                in which case only experiences collected using the latest policy version will be returned.
            proxy_kwargs: Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
                for details. Defaults to the empty dictionary.
            log_dir (str): Directory to store logs in. Defaults to the current working directory.
        """
        peers = {"actor": num_actors}
        name = "POLICY_SERVER"
        proxy = Proxy(group, "policy_server", peers, component_name=name, **proxy_kwargs)
        logger = Logger(name, dump_folder=log_dir)

        num_active_actors = num_actors
        for msg in proxy.receive():
            if msg.tag == MsgTag.GET_INITIAL_POLICY_STATE:
                proxy.reply(
                    msg, tag=MsgTag.POLICY_STATE,
                    body={MsgKey.POLICY_STATE: self.get_state(), MsgKey.VERSION: self.get_version()}
                )
            elif msg.tag == MsgTag.SAMPLE_DONE:
                if self.get_version() - msg.body[MsgKey.VERSION] > max_lag:
                    logger.info(
                        f"Ignored a message because it contains experiences generated using a stale policy version. "
                        f"Expected experiences generated using policy versions no earlier than "
                        f"{self.get_version() - max_lag}, got {msg.body[MsgKey.VERSION]}"
                    )
                else:
                    self.update(msg.body[MsgKey.ROLLOUT_INFO])
                proxy.reply(
                    msg, tag=MsgTag.POLICY_STATE,
                    body={MsgKey.POLICY_STATE: self.get_state(), MsgKey.VERSION: self.get_version()}
                )
            elif msg.tag == MsgTag.DONE:
                num_active_actors -= 1
                if num_active_actors == 0:
                    proxy.close()
                    return


class SimplePolicyManager(AbsPolicyManager):
    """Policy manager that contains all policy instances.

    Args:
        create_policy_func_dict (dict): Dictionary that maps policy names to policy creators. A policy creator is a
            function that takes policy name as the only parameter and return an ``RLPolicy`` instance.
        load_path_dict (Dict[str, str]): If provided, policies whose IDs are in the dictionary keys will load the states
            from the corresponding path. Defaults to None.
        checkpoint_every (int): The policies will be checkpointed (i.e., persisted to disk) every this number of seconds
            only if there are updates since the last checkpoint. This must be a positive integer or -1, with -1 meaning
            no checkpointing. Defaults to -1.
        save_dir (str): The directory under which to checkpoint the policy states.
        group (str): Group name for the cluster consisting of the manager and all policy hosts. It will be meaningless
            if ``data_parallel`` is False. Defaults to "learn".
        data_parallel (bool): If True, the policies will send gradient tasks to remote gradient workers for parallel
            gradient computation. Defaults to False.
        num_grad_workers (int): Number of gradient workers. Ignored if ``data_parallel`` is False. Defaults to 1.
        worker_allocator (WorkerAllocator): Strategy to select gradient workers for policies for send gradient tasks to.
        proxy_kwargs (dict): Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class for details.
        log_dir (str): Directory to store logs in. A ``Logger`` with tag "POLICY_MANAGER" will be created at init
            time and this directory will be used to save the log files generated by it. Defaults to the current
            working directory.
    """
    def __init__(
        self,
        create_policy_func_dict: Dict[str, Callable[[str], RLPolicy]],
        load_path_dict: Dict[str, str] = None,
        checkpoint_every: int = -1,
        save_dir: str = None,
        group: str = "learn",
        data_parallel: bool = False,
        num_grad_workers: int = 1,
        worker_allocator: WorkerAllocator = None,
        proxy_kwargs: dict = {},
        log_dir: str = getcwd()
    ):
        assert checkpoint_every == -1 or checkpoint_every > 0, "'checkpoint_every' can only be -1 or a positive integer"
        super().__init__()
        self._logger = Logger("POLICY_MANAGER", dump_folder=log_dir)
        self._policy_dict = {name: func(name) for name, func in create_policy_func_dict.items()}
        for id_, path in load_path_dict.items():
            self._policy_dict[id_].load(path)
            self._logger.info(f"Loaded policy {id_} from {path}")

        self._version = defaultdict(int)

        # auto-checkpointing
        self.checkpoint_every = checkpoint_every
        self.save_path = {id_: os.path.join(save_dir, id_) for id_ in create_policy_func_dict}
        if self.checkpoint_every > 0:
            checkpoint_thread = threading.Thread(target=self._checkpoint, daemon=True)
            checkpoint_thread.start()
            self._logger.info(f"Auto-checkpoint policy state every {self.checkpoint_every} seconds")

        # data parallelism
        self._data_parallel = data_parallel
        if self._data_parallel:
            self._num_grad_workers = num_grad_workers
            self._worker_allocator = worker_allocator
            self._logger = Logger("POLICY_MANAGER", dump_folder=log_dir)
            self._worker_allocator.set_logger(self._logger)
            self._proxy = Proxy(
                group, "policy_manager", {"grad_worker": self._num_grad_workers},
                component_name="POLICY_MANAGER", **proxy_kwargs)

            for name in create_policy_func_dict:
                self._policy_dict[name].data_parallel(
                    group, "policy_host", {"grad_worker": self._num_grad_workers},
                    component_name=f"POLICY_HOST.{name}", **proxy_kwargs)

    def update(self, rollout_info: Dict[str, list]):
        """Update policies using roll-out information.

        The roll-out information is grouped by policy name and may be either a training batch or a list of loss
        information dictionaries computed directly by roll-out workers.
        """
        t0 = time.time()
        for policy_id, info in rollout_info.items():
            if isinstance(info, list):
                self._policy_dict[policy_id].update(info)
            else:
                self._policy_dict[policy_id].learn(info)
            self._version[policy_id] += 1

        self._logger.info(f"Updated policies {list(rollout_info.keys())}")
        self._logger.info(f"policy update time: {time.time() - t0}")

    def get_state(self):
        """Get the latest policy states."""
        return {name: policy.get_state() for name, policy in self._policy_dict.items()}

    def get_version(self):
        """Get the collective policy version."""
        return dict(self._version)

    def save(self):
        for id_, policy in self._policy_dict.items():
            policy.save(self.save_path[id_])
            self._logger.info(f"Saved policy {id_} to {self.save_path[id_]}")

    def _checkpoint(self):
        last_checkpointed_version = defaultdict(lambda: -1)
        while True:
            for id_, policy in self._policy_dict.items():
                if self._version[id_] > last_checkpointed_version[id_]:
                    policy.save(self.save_path[id_])
                    self._logger.info(f"Saved policy {id_} to {self.save_path[id_]}")
                    last_checkpointed_version[id_] = self._version[id_]
                else:
                    self._logger.info(f"Latest version of policy {id_} already checkpointed")
            time.sleep(self.checkpoint_every)

    def exit(self):
        pass


class MultiProcessPolicyManager(AbsPolicyManager):
    """Policy manager that places each policy instance in a separate process.

    Args:
        create_policy_func_dict (dict): Dictionary that maps policy names to policy creators. A policy creator is a
            function that takes policy name as the only parameter and return an ``RLPolicy`` instance.
        load_path_dict (Dict[str, str]): If provided, policies whose IDs are in the dictionary keys will load the states
            from the corresponding path. Defaults to None.
        auto_checkpoint (bool): If True, the policies will be checkpointed (i.e., persisted to disk) every time they are
            updated. Defaults to -1.
        save_dir (str): The directory under which to checkpoint the policy states.
        group (str): Group name for the cluster consisting of the manager and all policy hosts. It will be meaningless
            if ``data_parallel`` is False. Defaults to "learn".
        data_parallel (bool): If True, the policies will send gradient tasks to remote gradient workers for parallel
            gradient computation. Defaults to False.
        num_grad_workers (int): Number of gradient workers. Ignored if ``data_parallel`` is False. Defaults to 1.
        worker_allocator (WorkerAllocator): The allocation strategy of allocating workers to policies
            for parallelization.
        proxy_kwargs (dict): Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class for details.
        log_dir (str): Directory to store logs in. A ``Logger`` with tag "POLICY_MANAGER" will be created at init
            time and this directory will be used to save the log files generated by it. Defaults to the current
            working directory.
    """
    def __init__(
        self,
        create_policy_func_dict: Dict[str, Callable[[str], RLPolicy]],
        load_path_dict: Dict[str, str] = None,
        auto_checkpoint: bool = False,
        save_dir: str = None,
        group: str = "learn",
        data_parallel: bool = False,
        num_grad_workers: int = 1,
        worker_allocator: WorkerAllocator = None,
        proxy_kwargs: dict = {},
        log_dir: str = getcwd()
    ):
        super().__init__()
        self.auto_checkpoint = auto_checkpoint
        self._logger = Logger("POLICY_MANAGER", dump_folder=log_dir)

        # data parallelism
        self._data_parallel = data_parallel
        if self._data_parallel:
            self._num_grad_workers = num_grad_workers
            self._worker_allocator = worker_allocator
            self._worker_allocator.set_logger(self._logger)
            self._proxy = Proxy(
                group, "policy_manager", {"grad_worker": self._num_grad_workers},
                component_name="POLICY_MANAGER", **proxy_kwargs
            )

        self._state_cache = {}
        self._policy_hosts = []
        self._manager_end = {}
        self._logger.info("Spawning policy host processes")

        def policy_host(id_, create_policy_func, conn, initial_state_path):
            policy = create_policy_func(id_)
            save_path = os.path.join(save_dir, id_)
            if self._data_parallel:
                self._logger.info("========== data parallel mode ==========")
                policy.data_parallel(
                    group, "policy_host", {"grad_worker": self._num_grad_workers}, component_name=f"POLICY_HOST.{id_}",
                    **proxy_kwargs)

            if initial_state_path:
                policy.load(initial_state_path)
                self._logger.info(f"Loaded policy {id_} from {initial_state_path}")
            conn.send({"type": "init", "policy_state": policy.get_state()})
            while True:
                msg = conn.recv()
                if msg["type"] == "learn":
                    info = msg["rollout_info"]
                    if isinstance(info, list):
                        policy.update(info)
                    elif self._data_parallel:
                        policy.learn_with_data_parallel(info, msg["workers"])
                    else:
                        policy.learn(info)
                    conn.send({"type": "learn_done", "policy_state": policy.get_state()})
                    self._logger.info("learning finished")
                    if self.auto_checkpoint:
                        policy.save(save_path)
                        self._logger.info(f"Saved policy {id_} to {save_path}")
                elif msg["type"] == "quit":
                    if self._data_parallel:
                        policy.exit_data_parallel()
                    policy.save(save_path)
                    self._logger.info(f"Saved policy {id_} to {save_path}")
                    break

        for id_, create_policy_func in create_policy_func_dict.items():
            manager_end, host_end = Pipe()
            self._manager_end[id_] = manager_end
            host = Process(
                target=policy_host, args=(id_, create_policy_func, host_end, load_path_dict.get(id_, None))
            )
            self._policy_hosts.append(host)
            host.start()

        for policy_id, conn in self._manager_end.items():
            msg = conn.recv()
            if msg["type"] == "init":
                self._state_cache[policy_id] = msg["policy_state"]
                self._logger.info(f"Initial state for policy {policy_id} cached")

        self._version = defaultdict(int)

    def update(self, rollout_info: Dict[str, list]):
        """Update policies using roll-out information.

        The roll-out information is grouped by policy name and may be either a training batch or a list of loss
        information dictionaries computed directly by roll-out workers.
        """
        if self._data_parallel:
            # re-allocate grad workers before update.
            self._policy2workers, self._worker2policies = self._worker_allocator.allocate()

        for policy_id, info in rollout_info.items():
            msg = {"type": "learn", "rollout_info": info}
            if self._data_parallel:
                msg["workers"] = self._policy2workers[policy_id]
            self._manager_end[policy_id].send(msg)
        for policy_id, conn in self._manager_end.items():
            msg = conn.recv()
            if msg["type"] == "learn_done":
                self._state_cache[policy_id] = msg["policy_state"]
                self._logger.info(f"Cached state for policy {policy_id}")
                self._version[policy_id] += 1
            else:
                self._logger.info(f"Warning: Wrong message type: {msg['type']}")

        self._logger.info(f"Updated policies {list(rollout_info.keys())}")

    def get_state(self):
        """Get the latest policy states."""
        return self._state_cache

    def get_version(self):
        """Get the collective policy version."""
        return dict(self._version)

    def exit(self):
        """Tell the policy host processes to exit."""
        for conn in self._manager_end.values():
            conn.send({"type": "quit"})
        if self._data_parallel:
            self._proxy.close()


class DistributedPolicyManager(AbsPolicyManager):
    """Policy manager that communicates with a set of remote nodes that house the policy instances.

    Args:
        policy_ids (List[str]): Names of the registered policies.
        group (str): Group name for the cluster consisting of the manager and all policy hosts.
        num_hosts (int): Number of hosts. The hosts will be identified by "POLICY_HOST.i", where 0 <= i < num_hosts.
        data_parallel (bool): If True, the policies will send gradient tasks to remote gradient workers for parallel
            gradient computation. Defaults to False.
        num_grad_workers (int): Number of gradient workers. Ignored if ``data_parallel`` is False. Defaults to 1.
        worker_allocator (WorkerAllocator): The allocation strategy of allocating workers to policies
            for parallelization.
        proxy_kwargs: Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to the empty dictionary.
        log_dir (str): Directory to store logs in. A ``Logger`` with tag "POLICY_MANAGER" will be created at init
            time and this directory will be used to save the log files generated by it. Defaults to the current
            working directory.
    """
    def __init__(
        self,
        policy_ids: List[str],
        group: str,
        num_hosts: int,
        data_parallel: bool = False,
        num_grad_workers: int = 1,
        worker_allocator: WorkerAllocator = None,
        proxy_kwargs: dict = {},
        log_dir: str = getcwd()
    ):
        super().__init__()
        peers = {"policy_host": num_hosts}
        if data_parallel:
            peers["grad_worker"] = num_grad_workers
        self._proxy = Proxy(group, "policy_manager", peers, component_name="POLICY_MANAGER", **proxy_kwargs)
        self._logger = Logger("POLICY_MANAGER", dump_folder=log_dir)
        self._worker_allocator = worker_allocator
        self._data_parallel = data_parallel

        self._worker_allocator.set_logger(self._logger)

        self._policy2host = {}
        self._policy2workers = {}
        self._host2policies = defaultdict(list)
        self._worker2policies = defaultdict(list)

        # assign policies to hosts
        for i, name in enumerate(policy_ids):
            host_id = i % num_hosts
            self._policy2host[name] = f"POLICY_HOST.{host_id}"
            self._host2policies[f"POLICY_HOST.{host_id}"].append(name)

        self._logger.info(f"Policy assignment: {self._policy2host}")

        # ask the hosts to initialize the assigned policies
        for host_name, policy_ids in self._host2policies.items():
            self._proxy.isend(SessionMessage(
                MsgTag.INIT_POLICIES, self._proxy.name, host_name, body={MsgKey.POLICY_IDS: policy_ids}
            ))

        # cache the initial policy states
        self._state_cache, dones = {}, 0
        for msg in self._proxy.receive():
            if msg.tag == MsgTag.INIT_POLICIES_DONE:
                for policy_id, policy_state in msg.body[MsgKey.POLICY_STATE].items():
                    self._state_cache[policy_id] = policy_state
                    self._logger.info(f"Cached state for policy {policy_id}")
                dones += 1
                if dones == num_hosts:
                    break

        self._version = 0

    def update(self, rollout_info: Dict[str, list]):
        """Update policies using roll-out information.

        The roll-out information is grouped by policy name and may be either a training batch or a list if loss
        information dictionaries computed directly by roll-out workers.
        """
        if self._data_parallel:
            self._policy2workers, self._worker2policies = self._worker_allocator.allocate()

        msg_dict = defaultdict(lambda: defaultdict(dict))
        for policy_id, info_list in rollout_info.items():
            host_id_str = self._policy2host[policy_id]
            msg_dict[host_id_str][MsgKey.ROLLOUT_INFO][policy_id] = info_list
            msg_dict[host_id_str][MsgKey.WORKER_INFO][policy_id] = self._policy2workers

        dones = 0
        self._proxy.iscatter(MsgTag.LEARN, SessionType.TASK, list(msg_dict.items()))
        for msg in self._proxy.receive():
            if msg.tag == MsgTag.LEARN_DONE:
                for policy_id, policy_state in msg.body[MsgKey.POLICY_STATE].items():
                    self._state_cache[policy_id] = policy_state
                    self._logger.info(f"Cached state for policy {policy_id}")
                dones += 1
                if dones == len(msg_dict):
                    break

        self._version += 1
        self._logger.info(f"Updated policies {list(rollout_info.keys())}")

    def get_state(self):
        """Get the latest policy states."""
        return self._state_cache

    def get_version(self):
        """Get the collective policy version."""
        return dict(self._version)

    def exit(self):
        """Tell the remote policy hosts to exit."""
        self._proxy.ibroadcast("policy_host", MsgTag.EXIT, SessionType.NOTIFICATION)
        self._proxy.close()
        self._logger.info("Exiting...")


def policy_host(
    create_policy_func_dict: Dict[str, Callable[[str], RLPolicy]],
    host_idx: int,
    group: str,
    proxy_kwargs: dict = {},
    data_parallel: bool = False,
    num_grad_workers: int = 1,
    log_dir: str = getcwd()
):
    """Policy host process that can be launched on separate computation nodes.

    Args:
        create_policy_func_dict (dict): A dictionary mapping policy names to functions that create them. The policy
            creation function should have policy name as the only parameter and return an ``RLPolicy`` instance.
        host_idx (int): Integer host index. The host's ID in the cluster will be "POLICY_HOST.{host_idx}".
        group (str): Group name for the training cluster, which includes all policy hosts and a policy manager that
            manages them.
        proxy_kwargs: Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to the empty dictionary.
        data_parallel (bool): Whether to train policy on remote gradient workers to perform data-parallel.
            Defaluts to False.
        num_grad_workers (int): The number of gradient worker nodes in data-parallel mode. Defaluts to 1.
        log_dir (str): Directory to store logs in. Defaults to the current working directory.
    """
    policy_dict = {}
    peers = {"policy_manager": 1}
    if data_parallel:
        peers["grad_worker"] = num_grad_workers

    proxy = Proxy(
        group, "policy_host", peers,
        component_name=f"POLICY_HOST.{host_idx}", **proxy_kwargs)
    logger = Logger(proxy.name, dump_folder=log_dir)

    for msg in proxy.receive():
        if msg.tag == MsgTag.EXIT:
            logger.info("Exiting...")
            proxy.close()
            break
        elif msg.tag == MsgTag.INIT_POLICIES:
            for name in msg.body[MsgKey.POLICY_IDS]:
                policy_dict[name] = create_policy_func_dict[name](name)
                if data_parallel:
                    policy_dict[name].data_parallel_with_existing_proxy(proxy)

            logger.info(f"Initialized policies {msg.body[MsgKey.POLICY_IDS]}")
            proxy.reply(
                msg,
                tag=MsgTag.INIT_POLICIES_DONE,
                body={MsgKey.POLICY_STATE: {name: policy.get_state() for name, policy in policy_dict.items()}}
            )
        elif msg.tag == MsgTag.LEARN:
            t0 = time.time()
            for name, info in msg.body[MsgKey.ROLLOUT_INFO].items():
                # in some cases e.g. Actor-Critic that get loss from rollout workers
                if isinstance(info, list):
                    logger.info("updating with loss info")
                    policy_dict[name].update(info)
                else:
                    if data_parallel:
                        logger.info("learning on remote grad workers")
                        policy2workers = msg.body[MsgKey.WORKER_INFO][name]
                        policy_dict[name].learn_with_data_parallel(info, policy2workers[name])
                    else:
                        logger.info("learning from batch")
                        policy_dict[name].learn(info)

            msg_body = {
                MsgKey.POLICY_STATE: {name: policy_dict[name].get_state() for name in msg.body[MsgKey.ROLLOUT_INFO]}
            }
            logger.info(f"total policy update time: {time.time() - t0}")
            proxy.reply(msg, tag=MsgTag.LEARN_DONE, body=msg_body)
        else:
            logger.info(f"Wrong message tag: {msg.tag}")
            raise TypeError


def grad_worker(
    create_policy_func_dict: Dict[str, Callable[[str], RLPolicy]],
    worker_idx: int,
    num_hosts: int,
    group: str,
    proxy_kwargs: dict = {},
    max_policy_number: int = 10,
    log_dir: str = getcwd()
):
    """Stateless gradient workers that excute gradient computation tasks.

    Args:
        create_policy_func_dict (dict): A dictionary mapping policy names to functions that create them. The policy
            creation function should have policy name as the only parameter and return an ``RLPolicy`` instance.
        worker_idx (int): Integer worker index. The worker's ID in the cluster will be "GRAD_WORKER.{worker_idx}".
        num_hosts (int): Number of policy hosts, which is required to find peers in proxy initialization.
            num_hosts=0 means policy hosts are hosted by policy manager while no remote nodes for them.
        group (str): Group name for the training cluster, which includes all policy hosts and a policy manager that
            manages them.
        proxy_kwargs: Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to the empty dictionary.
        max_policy_number (int): Maximum policy number in a single worker node. Defaults to 10.
        log_dir (str): Directory to store logs in. Defaults to the current working directory.
    """
    policy_dict = {}
    active_policies = []
    if num_hosts == 0:
        # no remote nodes for policy hosts
        num_hosts = len(create_policy_func_dict)
    peers = {"policy_manager": 1, "policy_host": num_hosts}
    proxy = Proxy(group, "grad_worker", peers, component_name=f"GRAD_WORKER.{worker_idx}", **proxy_kwargs)
    logger = Logger(proxy.name, dump_folder=log_dir)

    for msg in proxy.receive():
        if msg.tag == MsgTag.EXIT:
            logger.info("Exiting...")
            proxy.close()
            break
        elif msg.tag == MsgTag.COMPUTE_GRAD:
            t0 = time.time()
            msg_body = {MsgKey.LOSS_INFO: dict(), MsgKey.POLICY_IDS: list()}
            for name, batch in msg.body[MsgKey.GRAD_TASK].items():
                if name not in policy_dict:
                    if len(policy_dict) > max_policy_number:
                        # remove the oldest one when size exceeds.
                        policy_to_remove = active_policies.pop()
                        policy_dict.pop(policy_to_remove)
                    policy_dict[name] = create_policy_func_dict[name](name)
                    active_policies.insert(0, name)
                    logger.info(f"Initialized policies {name}")

                policy_dict[name].set_state(msg.body[MsgKey.POLICY_STATE][name])
                loss_info = policy_dict[name].get_batch_loss(batch, explicit_grad=True)
                msg_body[MsgKey.LOSS_INFO][name] = loss_info
                msg_body[MsgKey.POLICY_IDS].append(name)
                # put the latest one to queue head
                active_policies.remove(name)
                active_policies.insert(0, name)

            logger.debug(f"total policy update time: {time.time() - t0}")
            proxy.reply(msg, tag=MsgTag.COMPUTE_GRAD_DONE, body=msg_body)
        else:
            logger.info(f"Wrong message tag: {msg.tag}")
            raise TypeError