# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time

from maro.utils.exception.communication_exception import SendAgain


def peers_checker(func):
    def wrapper(self, *args, **kwargs):
        rejoin_start_time = time.time()
        while self._enable_rejoin:
            current_time = time.time()
            if current_time - rejoin_start_time > self._max_wait_time_for_rejoin:
                raise TimeoutError(f"Out of max waiting time for rejoin. Cannot reach the minimal number of peers.")

            if current_time - self._onboard_peers_lifetime > self._peer_update_frequency:
                self._onboard_peers_lifetime = current_time
                self._check_peers_update()

            if func.__name__ == "send" or func.__name__ == "isend":
                message = args[0] if args else kwargs["message"]
                peer_name = message.destination
                peer_type = peer_name.split("_proxy_")[0]
                # check message cache, if has, send pending message first.
                if self._enable_message_cache and peer_name in list(self._message_cache_for_exited_peers.keys()):
                    pending_session_ids = []
                    self._logger.warn(f"Sending pending message to {peer_name}.")
                    for pending_message in self._message_cache_for_exited_peers[peer_name]:
                        pending_session_ids.append(func(self, pending_message))
                    del self._message_cache_for_exited_peers[peer_name]

            output = func(self, *args, **kwargs)

            if isinstance(output, SendAgain):
                # Record messages for exited peers.
                if len(self._onboard_peers_name_dict[peer_type]) > self._minimal_peers[peer_type]:
                    if self._enable_message_cache:
                        self._message_cache_for_exited_peers.append(peer_name, message)
                    self._logger.critical(
                        f"Peer {peer_name} exited, but still have enough peers. Save message to message cache."
                    )
                    return output
                else:
                    self._logger.critical(
                        f"No enough peers! Waiting for peer {peer_name} restart. Remaining time: "
                        f"{rejoin_start_time + self._max_wait_time_for_rejoin - current_time}"
                    )
                    time.sleep(self._peer_update_frequency)
            else:
                if "pending_session_ids" in locals():
                    for recv in pending_session_ids:
                        output.append(recv[0])
                break

        return output if self._enable_rejoin else func(self, *args, **kwargs)

    return wrapper
