# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from multiprocessing import Pipe, Process
from os import getcwd, getpid
from typing import Callable, Dict, List

from maro.communication import Proxy, SessionMessage, SessionType
from maro.rl.policy import RLPolicy, WorkerAllocator
from maro.rl.utils import MsgKey, MsgTag
from maro.utils import Logger

# default group name for the cluster consisting of a policy manager and all policy hosts.
# If data parallelism is enabled, the gradient workers will also belong in this group.
DEFAULT_POLICY_GROUP = "policy_group_default"


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
                for details. Defaults to an empty dictionary.
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
        load_dir (str): If provided, policies whose IDs are in the dictionary keys will load the states
            from the corresponding path. Defaults to None.
        checkpoint_every (int): The policies will be checkpointed (i.e., persisted to disk) every this number of seconds
            only if there are updates since the last checkpoint. This must be a positive integer or -1, with -1 meaning
            no checkpointing. Defaults to -1.
        checkpoint_dir (str): The directory under which to checkpoint the policy states.
        worker_allocator (WorkerAllocator): Strategy to select gradient workers for policies for send gradient tasks to.
            If not None, the policies will be trained in data-parallel mode. Defaults to None.
        group (str): Group name for the cluster consisting of the manager and all policy hosts. Ignored if
            ``worker_allocator`` is None. Defaults to DEFAULT_POLICY_GROUP.
        proxy_kwargs (dict): Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class for details.
        log_dir (str): Directory to store logs in. A ``Logger`` with tag "POLICY_MANAGER" will be created at init
            time and this directory will be used to save the log files generated by it. Defaults to the current
            working directory.
    """
    def __init__(
        self,
        create_policy_func_dict: Dict[str, Callable[[str], RLPolicy]],
        load_dir: str = None,
        checkpoint_dir: str = None,
        worker_allocator: WorkerAllocator = None,
        group: str = DEFAULT_POLICY_GROUP,
        proxy_kwargs: dict = {},
        log_dir: str = getcwd()
    ):
        super().__init__()
        self._logger = Logger("POLICY_MANAGER", dump_folder=log_dir)
        self._policy_dict = {name: func(name) for name, func in create_policy_func_dict.items()}
        if load_dir:
            for id_, policy in self._policy_dict.items():
                path = os.path.join(load_dir, id_)
                if os.path.exists(path):
                    policy.load(path)
                    self._logger.info(f"Loaded policy {id_} from {path}")

        self._version = 0

        # auto-checkpointing
        if checkpoint_dir:
            self.checkpoint_path = {id_: os.path.join(checkpoint_dir, id_) for id_ in self._policy_dict}
        else:
            self.checkpoint_path = None

        # data parallelism
        self._worker_allocator = worker_allocator
        if self._worker_allocator:
            self._num_grad_workers = worker_allocator.num_workers
            self._worker_allocator.set_logger(self._logger)
            self._proxy = Proxy(
                group, "policy_manager", {"grad_worker": self._num_grad_workers},
                component_name="POLICY_MANAGER", **proxy_kwargs
            )

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

            if self.checkpoint_path:
                self._policy_dict[policy_id].save(self.checkpoint_path[policy_id])
                self._logger.info(f"Saved policy {policy_id} to {self.checkpoint_path[policy_id]}")

            self._version += 1

        self._logger.info(f"Updated policies {list(rollout_info.keys())}")
        self._logger.info(f"policy update time: {time.time() - t0}")

    def get_state(self):
        """Get the latest policy states."""
        return {name: policy.get_state() for name, policy in self._policy_dict.items()}

    def get_version(self):
        """Get the collective policy version."""
        return self._version

    def exit(self):
        pass


class MultiProcessPolicyManager(AbsPolicyManager):
    """Policy manager that places each policy instance in a separate process.

    Args:
        create_policy_func_dict (dict): Dictionary that maps policy names to policy creators. A policy creator is a
            function that takes policy name as the only parameter and return an ``RLPolicy`` instance.
        load_dir (str): If provided, policies whose IDs are in the dictionary keys will load the states
            from the corresponding path. Defaults to None.
        checkpoint_dir (str): The directory under which to checkpoint the policy states. Defaults to None, in which case
            no checkpointing will be performed.
        worker_allocator (WorkerAllocator): Strategy to select gradient workers for policies for send gradient tasks to.
            If not None, the policies will be trained in data-parallel mode. Defaults to None.
        group (str): Group name for the cluster consisting of the manager and all policy hosts. Ignored if
            ``worker_allocator`` is None. Defaults to DEFAULT_POLICY_GROUP.
        proxy_kwargs (dict): Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class for details.
        log_dir (str): Directory to store logs in. A ``Logger`` with tag "POLICY_MANAGER" will be created at init
            time and this directory will be used to save the log files generated by it. Defaults to the current
            working directory.
    """
    def __init__(
        self,
        create_policy_func_dict: Dict[str, Callable[[str], RLPolicy]],
        load_dir: Dict[str, str] = None,
        checkpoint_dir: str = None,
        worker_allocator: WorkerAllocator = None,
        group: str = DEFAULT_POLICY_GROUP,
        proxy_kwargs: dict = {},
        log_dir: str = getcwd()
    ):
        super().__init__()
        self._logger = Logger("POLICY_MANAGER", dump_folder=log_dir)

        # data parallelism
        self._worker_allocator = worker_allocator is not None
        if self._worker_allocator:
            self._num_grad_workers = worker_allocator.num_workers
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

        def policy_host(id_, create_policy_func, conn):
            self._logger.info(f"Host for policy {id_} started with PID {getpid()}")
            policy = create_policy_func(id_)
            checkpoint_path = os.path.join(checkpoint_dir, id_) if checkpoint_dir else None
            if self._worker_allocator:
                self._logger.info("========== data parallel mode ==========")
                policy.data_parallel(
                    group, "policy_host", {"grad_worker": self._num_grad_workers}, component_name=f"POLICY_HOST.{id_}",
                    **proxy_kwargs)

            if load_dir:
                load_path = os.path.join(load_dir, id_) if load_dir else None
                if os.path.exists(load_path):
                    policy.load(load_path)
                    self._logger.info(f"Loaded policy {id_} from {load_path}")

            conn.send({"type": "init", "policy_state": policy.get_state()})
            while True:
                msg = conn.recv()
                if msg["type"] == "learn":
                    info = msg["rollout_info"]
                    if isinstance(info, list):
                        policy.update(info)
                    elif self._worker_allocator:
                        policy.learn_with_data_parallel(info, msg["workers"])
                    else:
                        policy.learn(info)
                    conn.send({"type": "learn_done", "policy_state": policy.get_state()})
                    self._logger.info("learning finished")
                    if checkpoint_path:
                        policy.save(checkpoint_path)
                        self._logger.info(f"Saved policy {id_} to {checkpoint_path}")
                elif msg["type"] == "quit":
                    if self._worker_allocator:
                        policy.exit_data_parallel()
                    policy.save(checkpoint_path)
                    self._logger.info(f"Saved policy {id_} to {checkpoint_path}")
                    break

        for id_, create_policy_func in create_policy_func_dict.items():
            manager_end, host_end = Pipe()
            self._manager_end[id_] = manager_end
            host = Process(target=policy_host, args=(id_, create_policy_func, host_end))
            self._policy_hosts.append(host)
            host.start()

        for policy_id, conn in self._manager_end.items():
            msg = conn.recv()
            if msg["type"] == "init":
                self._state_cache[policy_id] = msg["policy_state"]
                self._logger.info(f"Initial state for policy {policy_id} cached")

        self._version = 0

    def update(self, rollout_info: Dict[str, list]):
        """Update policies using roll-out information.

        The roll-out information is grouped by policy name and may be either a training batch or a list of loss
        information dictionaries computed directly by roll-out workers.
        """
        if self._worker_allocator:
            # re-allocate grad workers before update.
            self._policy2workers, self._worker2policies = self._worker_allocator.allocate()

        for policy_id, info in rollout_info.items():
            msg = {"type": "learn", "rollout_info": info}
            if self._worker_allocator:
                msg["workers"] = self._policy2workers[policy_id]
            self._manager_end[policy_id].send(msg)
        for policy_id, conn in self._manager_end.items():
            msg = conn.recv()
            if msg["type"] == "learn_done":
                self._state_cache[policy_id] = msg["policy_state"]
                self._logger.info(f"Cached state for policy {policy_id}")
                self._version += 1
            else:
                self._logger.info(f"Warning: Wrong message type: {msg['type']}")

        self._logger.info(f"Updated policies {list(rollout_info.keys())}")

    def get_state(self):
        """Get the latest policy states."""
        return self._state_cache

    def get_version(self):
        """Get the collective policy version."""
        return self._version

    def exit(self):
        """Tell the policy host processes to exit."""
        for conn in self._manager_end.values():
            conn.send({"type": "quit"})
        if self._worker_allocator:
            self._proxy.close()


class DistributedPolicyManager(AbsPolicyManager):
    """Policy manager that communicates with a set of remote nodes that house the policy instances.

    Args:
        policy_ids (List[str]): Names of the registered policies.
        num_hosts (int): Number of hosts. The hosts will be identified by "POLICY_HOST.i", where 0 <= i < num_hosts.
        group (str): Group name for the cluster consisting of the manager and all policy hosts. If ``worker_allocator``
            is provided, the gradient workers will also belong to the same cluster. Defaults to
            DEFAULT_POLICY_GROUP.
        worker_allocator (WorkerAllocator): Strategy to select gradient workers for policies for send gradient tasks to.
            If not None, the policies will be trained in data-parallel mode. Defaults to None.
        proxy_kwargs: Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to an empty dictionary.
        log_dir (str): Directory to store logs in. A ``Logger`` with tag "POLICY_MANAGER" will be created at init
            time and this directory will be used to save the log files generated by it. Defaults to the current
            working directory.
    """
    def __init__(
        self,
        policy_ids: List[str],
        num_hosts: int,
        group: str = DEFAULT_POLICY_GROUP,
        worker_allocator: WorkerAllocator = None,
        proxy_kwargs: dict = {},
        log_dir: str = getcwd()
    ):
        super().__init__()
        self._worker_allocator = worker_allocator
        peers = {"policy_host": num_hosts}
        if self._worker_allocator:
            peers["grad_worker"] = worker_allocator.num_workers
        self._proxy = Proxy(group, "policy_manager", peers, component_name="POLICY_MANAGER", **proxy_kwargs)
        self._logger = Logger("POLICY_MANAGER", dump_folder=log_dir)
        self._worker_allocator = worker_allocator

        if self._worker_allocator:
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
        if self._worker_allocator:
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
        return self._version

    def exit(self):
        """Tell the remote policy hosts to exit."""
        self._proxy.ibroadcast("policy_host", MsgTag.EXIT, SessionType.NOTIFICATION)
        self._proxy.close()
        self._logger.info("Exiting...")    
