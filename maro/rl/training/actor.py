# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from os import getcwd
from typing import Dict

from maro.communication import Proxy
from maro.rl.env_wrapper import AbsEnvWrapper
from maro.rl.exploration import AbsExploration
from maro.rl.policy import AbsPolicy
from maro.utils import Logger

from .message_enums import MsgKey, MsgTag


class Actor(object):
    """On-demand roll-out executor.

    Args:
        env (AbsEnvWrapper): An ``AbsEnvWrapper`` instance that wraps an ``Env`` instance with scenario-specific
            processing logic and stores transitions during roll-outs in a replay memory.
        policy (MultiAgentPolicy): Agent that interacts with the environment.
        group (str): Identifier of the group to which the actor belongs. It must be the same group name
            assigned to the learner (and decision clients, if any).
        proxy_kwargs: Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to None.
    """
    def __init__(
        self,
        env: AbsEnvWrapper,
        policy_dict: Dict[str, AbsPolicy],
        agent2policy: Dict[str, str],
        group: str,
        exploration_dict: Dict[str, AbsExploration] = None,
        agent2exploration: Dict[str, str] = None,
        eval_env: AbsEnvWrapper = None,
        log_dir: str = getcwd(),
        **proxy_kwargs
    ):
        self.env = env
        self.eval_env = eval_env if eval_env else self.env

        # mappings between agents and policies
        self.policy_dict = policy_dict
        self.agent2policy = agent2policy
        self.policy = {agent_id: policy_dict[policy_id] for agent_id, policy_id in self.agent2policy.items()}
        self.agent_groups_by_policy = defaultdict(list)
        for agent_id, policy_id in agent2policy.items():
            self.agent_groups_by_policy[policy_id].append(agent_id)

        # mappings between exploration schemes and agents
        self.exploration_dict = exploration_dict
        if exploration_dict:
            self.agent2exploration = agent2exploration
            self.exploration = {
                agent_id: self.exploration_dict[exploration_id]
                for agent_id, exploration_id in self.agent2exploration.items()
            }
            self.exploration_enabled = True
            self.agent_groups_by_exploration = defaultdict(list)
            for agent_id, exploration_id in agent2exploration.items():
                self.agent_groups_by_exploration[exploration_id].append(agent_id)

            for exploration_id, agent_ids in self.agent_groups_by_exploration.items():
                self.agent_groups_by_exploration[exploration_id] = tuple(agent_ids)

        self._proxy = Proxy(group, "actor", {"actor_manager": 1}, **proxy_kwargs)
        self._logger = Logger(self._proxy.name, dump_folder=log_dir)

    def run(self):
        for msg in self._proxy.receive():
            if msg.tag == MsgTag.EXIT:
                self._logger.info("Exiting...")
                break

            if msg.tag == MsgTag.COLLECT:
                episode_index, segment_index = msg.body[MsgKey.EPISODE_INDEX], msg.body[MsgKey.SEGMENT_INDEX]
                if self.env.state is None:
                    self._logger.info(f"Training episode {msg.body[MsgKey.EPISODE_INDEX]}")
                    if hasattr(self, "exploration_dict"):
                        exploration_params = {
                            agent_ids: self.exploration_dict[exploration_id].parameters
                            for exploration_id, agent_ids in self.agent_groups_by_exploration.items()
                        }
                        self._logger.debug(f"Exploration parameters: {exploration_params}")

                    self.env.reset()
                    self.env.start()  # get initial state

                # load policies
                self._load_policy_states(msg.body[MsgKey.POLICY])

                starting_step_index = self.env.step_index + 1
                steps_to_go = float("inf") if msg.body[MsgKey.NUM_STEPS] == -1 else msg.body[MsgKey.NUM_STEPS]
                while self.env.state and steps_to_go > 0:
                    if self.exploration_dict:      
                        action = {
                            id_:
                                self.exploration[id_](self.policy[id_].choose_action(st))
                                if id_ in self.exploration else self.policy[id_].choose_action(st)
                            for id_, st in self.env.state.items()
                        }
                    else:
                        action = {id_: self.policy[id_].choose_action(st) for id_, st in self.env.state.items()}
                    self.env.step(action)
                    steps_to_go -= 1

                self._logger.info(
                    f"Roll-out finished for ep {episode_index}, segment {segment_index}"
                    f"(steps {starting_step_index} - {self.env.step_index})"
                )
                return_info = {
                    MsgKey.ENV_END: not self.env.state,
                    MsgKey.EPISODE_INDEX: episode_index,
                    MsgKey.SEGMENT_INDEX: segment_index,
                    MsgKey.EXPERIENCES: self.env.get_experiences(),
                    MsgKey.NUM_STEPS: self.env.step_index - starting_step_index + 1
                }

                if msg.body[MsgKey.RETURN_ENV_METRICS]:
                    return_info[MsgKey.METRICS] = self.env.metrics
                if not self.env.state:
                    if self.exploration_dict:
                        for exploration in self.exploration_dict.values():
                            exploration.step()

                    return_info[MsgKey.TOTAL_REWARD] = self.env.total_reward
                self._proxy.reply(msg, tag=MsgTag.COLLECT_DONE, body=return_info)
            elif msg.tag == MsgTag.EVAL:
                ep = msg.body[MsgKey.EPISODE_INDEX]
                self._logger.info(f"Evaluation episode {ep}")
                self.eval_env.reset()
                self.eval_env.start()  # get initial state
                self._load_policy_states(msg.body[MsgKey.POLICY])
                while self.eval_env.state:
                    action = {id_: self.policy[id_].choose_action(st) for id_, st in self.eval_env.state.items()}
                    self.eval_env.step(action)

                return_info = {
                    MsgKey.METRICS: self.env.metrics,
                    MsgKey.TOTAL_REWARD: self.eval_env.total_reward,
                    MsgKey.EPISODE_INDEX: msg.body[MsgKey.EPISODE_INDEX]  
                }
                self._proxy.reply(msg, tag=MsgTag.EVAL_DONE, body=return_info)

    def _load_policy_states(self, policy_state_dict):
        for policy_id, policy_state in policy_state_dict.items():
            self.policy_dict[policy_id].set_state(policy_state)

        if policy_state_dict:
            self._logger.info(f"updated policies {list(policy_state_dict.keys())}")
