# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json

import redis


class RedisController:
    def __init__(self, host: str, port: int):
        self._redis = redis.Redis(host=host, port=port, encoding="utf-8", decode_responses=True)

    """Master Details Related."""

    def get_master_details(self, cluster_name: str) -> dict:
        return json.loads(
            self._redis.get(f"{cluster_name}:master_details")
        )

    """Node Details Related."""

    def get_node_details(self, cluster_name: str, node_name: str) -> dict:
        return json.loads(
            self._redis.hget(
                f"{cluster_name}:node_details",
                node_name
            )
        )

    def get_nodes_details(self, cluster_name: str) -> dict:
        nodes_details = self._redis.hgetall(
            f"{cluster_name}:node_details"
        )
        for node_name, node_details in nodes_details.items():
            nodes_details[node_name] = json.loads(node_details)
        return nodes_details

    def set_node_details(self, cluster_name: str, node_name: str, node_details: dict) -> None:
        self._redis.hset(
            f"{cluster_name}:node_details",
            node_name,
            json.dumps(node_details)
        )

    """Job Details Related."""

    def get_job_details(self, cluster_name: str, job_name: str) -> dict:
        return_str = self._redis.hget(
            f"{cluster_name}:job_details",
            job_name
        )
        return json.loads(return_str) if return_str is not None else None

    def get_jobs_details(self, cluster_name: str) -> dict:
        jobs_details = self._redis.hgetall(
            f"{cluster_name}:job_details",
        )
        for job_name, job_details in jobs_details.items():
            jobs_details[job_name] = json.loads(job_details)
        return jobs_details

    def set_job_details(self, cluster_name: str, job_name: str, job_details: dict) -> None:
        self._redis.hset(
            f"{cluster_name}:job_details",
            job_name,
            json.dumps(job_details)
        )

    """Containers Details Related."""

    def get_containers_details(self, cluster_name: str) -> dict:
        containers_details = self._redis.hgetall(
            f"{cluster_name}:container_details",
        )
        for container_name, container_details in containers_details.items():
            containers_details[container_name] = json.loads(container_details)
        return containers_details

    def set_containers_details(self, cluster_name: str, containers_details: dict) -> None:
        self._redis.delete(f"{cluster_name}:container_details")
        if len(containers_details) == 0:
            return
        else:
            for container_name, container_details in containers_details.items():
                containers_details[container_name] = json.dumps(container_details)
            self._redis.hmset(
                f"{cluster_name}:container_details",
                containers_details
            )

    def set_container_details(self, cluster_name: str, container_name: str, container_details: dict) -> None:
        self._redis.hset(
            f"{cluster_name}:container_details",
            container_name,
            container_details
        )

    """Tickets Related."""

    def get_pending_job_tickets(self, cluster_name: str):
        return self._redis.lrange(
            f"{cluster_name}:pending_job_tickets",
            0,
            -1
        )

    def remove_pending_job_ticket(self, cluster_name: str, job_name: str):
        self._redis.lrem(
            f"{cluster_name}:pending_job_tickets",
            0,
            job_name
        )

    def get_killed_job_tickets(self, cluster_name: str):
        return self._redis.lrange(
            f"{cluster_name}:killed_job_tickets",
            0,
            -1
        )

    def remove_killed_job_ticket(self, cluster_name: str, job_name: str):
        self._redis.lrem(
            f"{cluster_name}:killed_job_tickets",
            0,
            job_name
        )

    """Fault Tolerance Related"""

    def get_rejoin_component_name_to_container_name(self, job_id: str) -> dict:
        return self._redis.hgetall(
            f"job:{job_id}:rejoin_component_name_to_container_name"
        )

    def get_rejoin_container_name_to_component_name(self, job_id: str) -> dict:
        component_name_to_container_name = self.get_rejoin_component_name_to_container_name(job_id=job_id)
        return {v: k for k, v in component_name_to_container_name.items()}

    def delete_rejoin_container_name_to_component_name(self, job_id: str) -> None:
        self._redis.delete(
            f"job:{job_id}:rejoin_component_name_to_container_name"
        )

    def get_job_runtime_details(self, job_id: str) -> dict:
        return self._redis.hgetall(
            f"job:{job_id}:runtime_details"
        )

    def get_rejoin_component_restart_times(self, job_id: str, component_id: str) -> int:
        restart_times = self._redis.hget(
            f"job:{job_id}:component_id_to_restart_times",
            component_id
        )
        return 0 if restart_times is None else int(restart_times)

    def incr_rejoin_component_restart_times(self, job_id: str, component_id: str) -> None:
        self._redis.hincrby(
            f"job:{job_id}:component_id_to_restart_times",
            component_id,
            1
        )

    # Utils

    def get_time(self) -> int:
        """ Get current unix timestamp (seconds) from Redis server.

        Returns:
            int: current timestamp.
        """
        return self._redis.time()[0]
