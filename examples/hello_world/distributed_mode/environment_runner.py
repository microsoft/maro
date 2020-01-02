# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import time
import os

import numpy as np
# private lib
from maro.distributed import Proxy


class MockSimulator:
    def __init__(self):
        self.proxy = Proxy(receive_enabled=True, audience_list=['learner'], redis_host='localhost', redis_port=6379)

    def await_model_from_learner(self, ep):
        """
        Wait for the learner's model. If the received episode number matches the current
        episode number, proceed to the next episode
        """
        for msg in self.proxy.receive():
            print(f'Received a {msg.type} message from {msg.src}: {msg.body["model"]}')
            if msg.type == 'model' and msg.body['episode'] == ep:
                break

    def launch(self, group_name, component_name):
        """
        Run 3 mock episodes and send a check-out message to the learner in the end
        """
        self.proxy.join(group_name, component_name)
        for ep in range(3):
            print(f'Running episode {ep}')
            time.sleep(2)
            self.proxy.send(peer_name='learner', msg_type='experience',
                            msg_body={'episode': ep, 'experience': np.random.rand(5)})
            self.await_model_from_learner(ep)

        self.proxy.send(peer_name='learner', msg_type='check_out', msg_body={})


if __name__ == '__main__':
    env = MockSimulator()
    env.launch('hello_world', 'env_runner')
