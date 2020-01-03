# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import sys
import numpy as np

from maro.distributed import dist
from maro.distributed import Proxy

# create a proxy for communication
proxy = Proxy(receive_enabled=True, audience_list=['env_runner'], redis_host='localhost', redis_port=6379)

############################### start of message handler definitions ###############################


def on_new_experience(local_instance, proxy, msg):
    """
    Handles hello messages from the environment
    """
    print(f'Received a {msg.type} message from {msg.src}: {msg.body["experience"]}')
    proxy.send(peer_name=msg.src, msg_type='model',
               msg_body={'episode': msg.body['episode'], 'model': np.random.rand(3)})


def on_checkout(local_instance, proxy, msg):
    """
    Handles the check-out message from the environment
    """
    print(f'Received a {msg.type} message from {msg.src}. Byebye!')
    sys.exit()


handler_dict = {'experience': on_new_experience, 'check_out': on_checkout}


############################### end of message handler definitions ###############################


@dist(proxy=proxy, handler_dict=handler_dict)
class MockLearner:
    def __init__(self):
        pass


if __name__ == '__main__':
    learner = MockLearner()
    learner.launch('hello_world', 'learner')
