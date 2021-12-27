import asyncio
import time

import torch

from maro.rl_v3.distributed.remote import RemoteOps
from maro.rl_v3.workflows.test.ops_creator import ops_creator

DISPATCHER_ADDRESS = ("127.0.0.1", 10000)


class SingleTrainer:
    def __init__(self, name, ops_creator, dispatcher_address=None) -> None:
        self._name = name
        ops_name = [name for name in ops_creator if name.startswith(f"{self._name}.")].pop()
        if dispatcher_address:
            self.remote_ops = RemoteOps(ops_name, DISPATCHER_ADDRESS)
        else:
            self._ops = ops_creator["ops"]("ops")

    async def train(self, X, y):
        await asyncio.gather(self.remote_ops.step(X, y))


class MultiTrainer:
    def __init__(self, name, ops_creator, dispatcher_address=None) -> None:
        self._name = name
        ops_names = [name for name in ops_creator if name.startswith(f"{self._name}.")]
        if dispatcher_address:
            self._ops_list = [RemoteOps(ops_name, DISPATCHER_ADDRESS) for ops_name in ops_names]
        else:
            self._ops_list = [ops_creator[ops_name](ops_name) for ops_name in ops_names]

    async def train(self, X, y):
        for _ in range(3):
            await asyncio.gather(*[ops.step(X, y) for ops in self._ops_list])


single_trainer = SingleTrainer("single", ops_creator, dispatcher_address=DISPATCHER_ADDRESS)
multi_trainer = MultiTrainer("multi", ops_creator, dispatcher_address=DISPATCHER_ADDRESS)

X, y = torch.rand(10, 5), torch.randint(0, 2, (10,))
X2, y2 = torch.rand(15, 7), torch.randint(0, 3, (15,))

t0 = time.time()
async def train():
    return await asyncio.gather(
        single_trainer.train(X, y),
        multi_trainer.train(X2, y2)
    )
asyncio.run(train())
print(f"total_time: {time.time() - t0}")
