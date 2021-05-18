# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict, namedtuple
from os import getcwd
from typing import List, Union

from maro.utils import Logger

from .policy_manager import AbsPolicyManager
from .rollout_manager import AbsRolloutManager

InterEpisodeSchedule = namedtuple("InterEpisodeSchedule", ["start", "interval"])
IntraEpisodeSchedule = namedtuple("IntraEpisodeSchedule", ["start_ep", "interval", "end_ep_update"])


class Learner:
    """Learner class for distributed training.

    Args:
        policy (MultiAgentPolicy): Learning agents.
        
    """
    def __init__(
        self,
        policy_manager: AbsPolicyManager,
        rollout_manager: AbsRolloutManager,
        num_episodes: int,
        eval_schedule: Union[int, List[int]] = None,
        log_dir: str = getcwd(),
        **end_of_episode_kwargs
    ):
        self.logger = Logger("LEARNER", dump_folder=log_dir)
        self.policy_manager = policy_manager
        self.rollout_manager = rollout_manager

        self.num_episodes = num_episodes

        # self._init_update_schedule(policy_update_schedule)

        # evaluation schedule
        if eval_schedule is None:
            eval_schedule = []
        elif isinstance(eval_schedule, int):
            num_eval_schedule = num_episodes // eval_schedule
            eval_schedule = [eval_schedule * i for i in range(1, num_eval_schedule + 1)]

        self._eval_schedule = eval_schedule
        self._eval_schedule.sort()
        if not self._eval_schedule or num_episodes != self._eval_schedule[-1]:
            self._eval_schedule.append(num_episodes)

        self.logger.info(f"Policy will be evaluated at the end of episodes {self._eval_schedule}")
        self._eval_point_index = 0

        self._end_of_episode_kwargs = end_of_episode_kwargs
        self._updated_policy_ids = self.policy_manager.names
        self._last_step_set = {}

    def run(self):
        for ep in range(1, self.num_episodes + 1):
            self._train(ep)
            if ep == self._eval_schedule[self._eval_point_index]:
                self._eval_point_index += 1
                self.rollout_manager.evaluate(self._eval_point_index)

    def _train(self, ep: int):
        num_experiences_collected = 0
        segment = 0
        self.rollout_manager.reset()
        while not self.rollout_manager.episode_complete:
            segment += 1
            # experience collection
            policy_state_dict = self.policy_manager.get_state()
            exp_by_agent = self.rollout_manager.collect(ep, segment, policy_state_dict)
            self.policy_manager.on_experiences(exp_by_agent)
            num_experiences_collected += sum(exp.size for exp in exp_by_agent.values())

        # performance details
        self.logger.debug(f"ep {ep} summary - experiences collected: {num_experiences_collected}")

        self.end_of_episode(ep, **self._end_of_episode_kwargs)        

    def end_of_episode(self, ep: int, **kwargs):
        pass

    def _init_update_schedule(self, schedule):
        self._pending_policies_by_segment = defaultdict(set)
        self._pending_policies_by_episode = defaultdict(list)
        self._step_schedule_opt = {}
        if isinstance(schedule, dict):
            for policy_id, sch in schedule.items():
                if isinstance(sch, IntraEpisodeSchedule):
                    self._step_schedule_opt[policy_id] = (sch.start_ep, sch.interval)
                    if sch.end_ep_update:
                        for ep in range(sch.start_ep, self.num_episodes + 1):
                            self._pending_policies_by_episode[ep].append(policy_id)
                    self._pending_policies_by_segment[sch.interval].add(policy_id)
                elif isinstance(sch, InterEpisodeSchedule):
                    for ep in range(sch.start, self.num_episodes + 1, step=sch.interval):
                        self._pending_policies_by_episode[ep].append(policy_id)
        else:
            if isinstance(schedule, IntraEpisodeSchedule):
                self._step_schedule_opt["*"] = (schedule.start_ep, schedule.interval)
                self._pending_policies_by_segment[schedule.interval].add("*")
                if schedule.end_ep_update:
                    for ep in range(schedule.start_ep, self.num_episodes + 1):
                        self._pending_policies_by_episode[ep] = self.policy_manager.names
            else:
                for ep in range(schedule.start, self.num_episodes + 1, step=schedule.interval):
                    self._pending_policies_by_episode[ep] = self.policy_manager.names

    def _get_pending_policy_ids(self, ep: int, segment: int) -> List[str]:
        for policy_id in self._pending_policies_by_segment[segment]:
            if segment == self._last_step_set[policy_id]:
                next_segment = segment + self._step_schedule_opt[policy_id][1]
                self._pending_policies_by_segment[next_segment].append(policy_id)

        return [
            policy_id for policy_id in self._pending_policies_by_segment[segment] if self._step_schedule_opt[policy_id][0] <= ep
        ]
