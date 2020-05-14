# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import time
import os

import numpy as np
# private lib
from maro.distributed import Proxy, Message

EXP = os.environ.get('GROUP')
REDIS = os.environ.get('REDIS')
PAYLOAD_SIZE = os.environ.get("PAYLOAD_SIZE")
ENV_NUM = os.environ.get("ENV_NUM")
SEND_BY = os.environ.get("SEND_BY")

class MockSimulator:
    def __init__(self):
        self.peer_list = ['learner_' + str(i) for i in range(int(ENV_NUM))]
        self.proxy = Proxy(group_name=EXP, component_name='env_runner',
                           peer_name_list=self.peer_list, redis_address=(REDIS, 6379))

    def launch(self):
        """
        Run 3 mock episodes and send a check-out message to the learner in the end
        """
        self.proxy.join()
        # for ep in range(10):
        #     # print(f'Running episode {ep}')
        #     # time.sleep(2)
        #     message = Message(type='experience', source=self.proxy.name,
        #                       destination='learner', payload=np.random.rand(1000000))
            
        #     t = self.proxy.send(message, multithread=True)
        #     # print(t)
        payload = np.random.rand(int(PAYLOAD_SIZE))
        time.sleep(10)
        start=time.time()
        t = self.proxy.ibroadcast('experience', self.peer_list, payload, sendby=SEND_BY)
        get_msg = self.proxy.get(t)
        end=time.time()
        # self.await_model_from_learner()
        print(str(end-start))
        for dest in self.peer_list:
            message = Message(type='check_out', source=self.proxy.name,
                            destination=dest)
            self.proxy.send(message, sendby='PUSH')
        time.sleep(100)


if __name__ == '__main__':
    env = MockSimulator()
    env.launch()
