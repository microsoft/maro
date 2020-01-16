# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import numpy as np

from maro.distributed import dist
from maro.distributed import Proxy, Message

# create a proxy for communication
proxy = Proxy(group_name='hello_world', component_name='learner',
              peer_name_list=['env_runner'], redis_address=('localhost', 6379))

############################### start of message handler definitions ###############################


def on_new_experience(local_instance, proxy, message):
    """
    Handles hello messages from the environment
    """
    print(f'Received a {message.type} message from {message.source}: {message.payload}')
    message = Message(type='model', source=proxy.name,
                      destination=message.source, payload=np.random.rand(3))
    proxy.send(message)


def on_checkout(local_instance, proxy, message):
    """
    Handles the check-out message from the environment
    """
    print(f'Received a {message.type} message from {message.source}. Byebye!')
    sys.exit()


# mock learner handles two message types: 'experience' and 'check_out'
handler_dict = {'experience': on_new_experience, 'check_out': on_checkout}

############################### end of message handler definitions ###############################


@dist(proxy=proxy, handler_dict=handler_dict)
class MockLearner:
    def __init__(self):
        pass


if __name__ == '__main__':
    learner = MockLearner()
    learner.launch()
