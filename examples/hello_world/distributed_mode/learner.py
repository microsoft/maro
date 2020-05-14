# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import numpy as np

from maro.distributed import dist
from maro.distributed import Proxy, Message

LR_INDEX = os.environ.get('LR_INDEX')
COMPONENT_NAME = 'learner_' + str(LR_INDEX)
EXP = os.environ.get('GROUP')
REDIS = os.environ.get('REDIS')

# create a proxy for communication
proxy = Proxy(group_name=EXP, component_name=COMPONENT_NAME,
              peer_name_list=['env_runner'], redis_address=(REDIS, 6379))

############################### start of message handler definitions ###############################

def on_new_experience(local_instance, proxy, list_message):
    """
    Handles hello messages from the environment
    """
    message = list_message[0]
    print(f'Received a {message.type} message from {message.source}: {message.payload}')
    send_message = Message(type='model', source=proxy.name,
                            destination=message.source, payload=np.random.rand(3),
                            message_id=message.message_id)
    proxy.send(send_message, 'PUSH')


def on_checkout(local_instance, proxy, list_message):
    """
    Handles the check-out message from the environment
    """
    message = list_message[0]
    print(f'Received a {message.type} message from {message.source}. Byebye!')
    sys.exit()


# mock learner handles two message types: 'experience' and 'check_out'
# handler_dict = {'experience': on_new_experience, 'check_out': on_checkout}
handler_dict = [{'request': {(True, 'experience'):1},
                 'handler_fn': on_new_experience},
                {'request': {(True, 'check_out'):1},
                 'handler_fn': on_checkout}]

############################### end of message handler definitions ###############################


@dist(proxy=proxy, handler_dict=handler_dict)
class MockLearner:
    def __init__(self):
        pass


if __name__ == '__main__':
    learner = MockLearner()
    learner.launch()
