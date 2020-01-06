# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import time

import numpy as np
# private lib
from maro.distributed import Proxy, Message


class MockSimulator:
    def __init__(self):
        self.proxy = Proxy(group_name='hello_world', component_name='env_runner',
                           peer_name_list=['learner'], redis_address=('localhost', 6379))

    def await_model_from_learner(self):
        """
        Wait for the learner's model.
        """
        msg = self.proxy.receive_once()
        print(f'Received a {msg.type} message from {msg.source}: {msg.body}')

    def launch(self):
        """
        Run 3 mock episodes and send a check-out message to the learner in the end
        """
        self.proxy.join()
        for ep in range(3):
            print(f'Running episode {ep}')
            time.sleep(2)
            message = Message(type_='experience', source=self.proxy.name,
                              destination='learner', body=np.random.rand(5))
            self.proxy.send(message)
            self.await_model_from_learner()

        message = Message(type_='check_out', source=self.proxy.name,
                          destination='learner')
        self.proxy.send(message)


if __name__ == '__main__':
    env = MockSimulator()
    env.launch()
