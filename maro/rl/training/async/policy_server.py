# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from os import getcwd
from typing import Dict, List

from maro.rl.policy import AbsPolicy
from maro.communication import Proxy
from maro.utils import Logger

from ..message_enums import MsgKey, MsgTag
from ..policy_manager import AbsPolicyManager


def policy_server(
    policy_manager: AbsPolicyManager,
    num_actors: int,
    group: str,
    num_requests_per_inference: int,
    proxy_kwargs: dict = {}
):
    peers = {"actor": num_actors}
    proxy = Proxy(group, "inference_server", peers, **proxy_kwargs)

    request_cache = {}
    state_batch_cache_by_policy = defaultdict(list)
    actor_offset = {policy_name: {} for policy_name in policy_manager.policy_dict}
    action_batch_by_actor = defaultdict(dict)

    for msg in proxy.receive():
        if msg.tag == MsgTag.CHOOSE_ACTION:
            # accumulate states for batch inference
            for policy_name, state_batch in msg.body[MsgKey.STATE].items():
                state_batch_cache_by_policy[policy_name].extend(state_batch)
                actor_offset[policy_name][msg.source] = len(state_batch_cache_by_policy[policy_name]) 

            request_cache[msg.source] = msg
            if len(request_cache) == num_requests_per_inference:
                for policy_name, state_batch in state_batch_cache_by_policy.items():
                    action_batch = policy_manager.policy_dict[policy_name].choose_action(state_batch)
                    last_index = 0
                    for actor_id, index in actor_offset[policy_name].items():
                        action_batch_by_actor[actor_id][policy_name] = action_batch[last_index:index]
                        last_index = index

                for actor_id, action_batch_by_policy in action_batch_by_actor.items():
                    request = request_cache[actor_id]
                    msg_body = {
                        MsgKey.EPISODE: request.body[MsgKey.EPISODE],
                        MsgKey.STEP: request.body[MsgKey.STEP],
                        MsgKey.ACTION: action_batch_by_policy
                    }
                    proxy.reply(request_cache[actor_id], tag=MsgTag.ACTION, body=msg_body)

                # reset internal data structures
                request_cache.clear()
                state_batch_cache_by_policy.clear()
                actor_offset = {policy_name: {} for policy_name in policy_manager.policy_dict}
                action_batch_by_actor.clear()
