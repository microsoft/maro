# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time


def peers_checker(func):
    def wrapper(self, *args, **kwargs):
        current_time = time.time()
        if current_time - self._onboard_peers_lifetime > self._peer_update_frequency and self._is_dynamic_peer:
            self._onboard_peers_lifetime = current_time
            self._update_peers()
            # Record messages for exited peers.
            if func.__name__ == "send" or func.__name__ == "isend":
                message = args[0] if args else kwargs["message"]
                peer_name = message.destination
                peer_type = peer_name.split("_proxy_")[0]

                if peer_name in self._exited_peer_dict[peer_type]:
                    self._message_cache_for_exited_peers[peer_name].append(message)

        return func(self, *args, **kwargs)

    return wrapper
