# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Union

from maro.communication import Message, Proxy
from maro.rl.agent import AbsAgent, MultiAgentWrapper
from maro.rl.storage import SimpleStore
from maro.rl.utils import get_sars
from maro.utils import InternalLogger

from .message_enums import MsgTag, MsgKey
from .trajectory import AbsTrajectory


class Actor(object):
    """Actor class that performs roll-out tasks.

    Args:
        trajectory (AbsTrajectory): An ``AbsTrajectory`` instance with an env wrapped inside.
        agent (Union[AbsAgent, MultiAgentWrapper]): Agent that interacts with the environment.
        group (str): Identifier of the group to which the actor belongs. It must be the same group name
            assigned to the learner (and decision clients, if any).
        proxy_options (dict): Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to None.
        trajectory_sync_interval (int): Number of roll-out steps between trajectory syncing calls.
        experience_getter (Callable): Custom function to extract experiences from a trajectory for training.
            If None, ``get_sars`` will be used. Defaults to None.
    """
    def __init__(
        self,
        trajectory: AbsTrajectory,
        agent: Union[AbsAgent, MultiAgentWrapper],
        group: str = None,
        proxy_options: dict = None,
        trajectory_sync_interval: int = None,
        experience_getter: Callable = get_sars
    ):
        self.trajectory = trajectory
        self.agent = MultiAgentWrapper(agent) if isinstance(agent, AbsAgent) else agent
        if group is not None:
            if proxy_options is None:
                proxy_options = {}
            self._proxy = Proxy(group, "actor", {"actor_proxy": 1}, **proxy_options)
            self._trajectory_sync_interval = trajectory_sync_interval
            self._logger = InternalLogger(self._proxy.name)

        # Under local mode or disributed mode without experience syncing, an experience pool needs to be created.
        if group is None or not self._trajectory_sync_interval:
            self.experience_pool = {
                agent_id: SimpleStore(["S", "A", "R", "S_", "loss"]) for agent_id in self.agent.agent_dict
            }
            self.experience_getter = experience_getter

    def roll_out(self, index: int, training: bool = True, model_by_agent: dict = None, exploration_params=None):
        self.trajectory.reset()
        if not training:
            self.trajectory.record_path = False  # no need to record the trajectory if roll-out is not for training

        # Load models and exploration parameters
        if model_by_agent:
            self.agent.load_model(model_by_agent)
        if exploration_params:
            self.agent.set_exploration_params(exploration_params)

        state = self.trajectory.start(rollout_index=index)  # get initial state
        while state:
            action = self.agent.choose_action(state)
            state = self.trajectory.step(action)
            if training and hasattr(self, "_proxy") and not hasattr(self, "experience_pool"): 
                self._sync_trajectory(index)
            self.trajectory.on_env_feedback()

        self.trajectory.on_finish()
        # If no experience syncing, the experience pool needs to be populated.
        if training and hasattr(self, "experience_pool"):
            for agent_id, exp in self.experience_getter(*self.trajectory.path).items():
                self.experience_pool[agent_id].put(exp)
                print(agent_id, len(self.experience_pool[agent_id]))

        return self.trajectory.env.metrics 

    def run(self):
        assert hasattr(self, "_proxy"), "No proxy found. The `group` parameter should not be None at init."
        for msg in self._proxy.receive():
            if msg.tag == MsgTag.EXIT:
                self._logger.info("Exiting...")
                break
            elif msg.tag == MsgTag.ROLLOUT:
                ep = msg.body[MsgKey.ROLLOUT_INDEX]
                self._logger.info(f"Rolling out ({ep})...")
                metrics = self.roll_out(
                    ep,
                    training=msg.body[MsgKey.TRAINING],
                    model_by_agent=msg.body[MsgKey.MODEL],
                    exploration_params=msg.body[MsgKey.EXPLORATION_PARAMS]
                )
                self._logger.info(f"Roll-out {ep} finished")
                body = {MsgKey.ROLLOUT_INDEX: ep, MsgKey.METRICS: metrics}
                if msg.body[MsgKey.TRAINING]:
                    if hasattr(self, "experience_pool"):
                        body[MsgKey.EXPERIENCE] = {id_: pool.data for id_, pool in self.experience_pool.items()}
                    else:
                        body[MsgKey.TRAJECTORY] = self.trajectory.path
                self._proxy.isend(
                    Message(MsgTag.ROLLOUT_DONE, self._proxy.name, self._proxy.peers["actor_proxy"][0], body=body)
                )

    def _sync_trajectory(self, index):
        if (self.trajectory.step_index + 1) % self._trajectory_sync_interval == 0:
            self._proxy.isend(
                Message(
                    MsgTag.TRAJECTORY_SYNC, self._proxy.name, self._proxy.peers["actor_proxy"][0],
                    body={MsgKey.ROLLOUT_INDEX: index, MsgKey.TRAJECTORY: self.trajectory.path}
                )
            )
            self.trajectory.flush()
