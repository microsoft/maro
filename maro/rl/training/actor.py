# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Union

from maro.communication import Message, Proxy
from maro.rl.agent import AbsAgent, MultiAgentWrapper
from maro.rl.storage import SimpleStore
from maro.rl.utils import get_sars
from maro.utils import InternalLogger

from .message_enums import MessageTag, PayloadKey
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
        sync_size (int): 
    """
    def __init__(
        self,
        trajectory: AbsTrajectory,
        agent: Union[AbsAgent, MultiAgentWrapper],
        group: str = None,
        proxy_options: dict = None,
        experience_sync_interval: int = None
    ):
        self.trajectory = trajectory
        self.agent = agent
        self.experience_pool = SimpleStore(["S", "A", "R", "S_", "loss"])
        if group is not None:
            if proxy_options is None:
                proxy_options = {}
            self._proxy = Proxy(group, "actor", {"actor_proxy": 1}, **proxy_options)
            self._experience_sync_interval = experience_sync_interval
            self._logger = InternalLogger(self._proxy.name)
        self.exp_sync = bool(getattr(self, "_experience_sync_interval", None))
        
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
            if training and self.exp_sync and len(self.trajectory.states) == self._experience_sync_interval:
                exp = get_sars(self.trajectory.states, self.trajectory.actions, self.trajectory.rewards)
                self.trajectory.flush()
                self._proxy.isend(
                    Message(
                        MessageTag.EXPERIENCE, self._proxy.name, self._proxy.peers_name["actor_proxy"][0],
                        payload={PayloadKey.ROLLOUT_INDEX: index, PayloadKey.EXPERIENCE: exp}
                    )
                )

            self.trajectory.on_env_feedback()

        self.trajectory.on_finish()
        # If no experience syncing, the experience pool needs to be populated.
        if training and not self._experience_sync_interval:
            if isinstance(self.agent, AbsAgent):
                self.experience_pool.put(get_sars(*self.trajectory.path, multi_agent=False))
            else: 
                for agent_id, exp in get_sars(*self.trajectory.path).items():
                    self.experience_pool[agent_id].put(exp)

        return self.trajectory.env.metrics 
    
    def run(self):
        assert hasattr(self, "_proxy"), "No proxy found. The `group` parameter should not be None at init."
        for msg in self._proxy.receive():
            if msg.tag == MessageTag.EXIT:
                self._logger.info("Exiting...")
                break
            elif msg.tag == MessageTag.ROLLOUT:
                ep = msg.payload[PayloadKey.ROLLOUT_INDEX]
                self._logger.info(f"Rolling out ({ep})...")
                metrics = self.roll_out(
                    ep,
                    training=msg.payload[PayloadKey.TRAINING],
                    model_by_agent=msg.payload[PayloadKey.MODEL],
                    exploration_params=msg.payload[PayloadKey.EXPLORATION_PARAMS]
                )
                self._logger.info(f"Roll-out {ep} finished")
                payload = {PayloadKey.ROLLOUT_INDEX: ep, PayloadKey.METRICS: metrics}
                if msg.payload[PayloadKey.TRAINING] and not self.exp_sync:
                    payload[PayloadKey.EXPERIENCE] = self.experience_pool
                self._proxy.isend(
                    Message(
                        MessageTag.FINISHED, self._proxy.name, self._proxy.peers_name["actor_proxy"][0],
                        payload=payload
                    )
                )
