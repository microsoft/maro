import asyncio
import time

import torch

from maro.rl_v3.training.trainer import BatchTrainer
from maro.rl_v3.utils.distributed import RemoteObj, CoroutineAdapter
from maro.rl_v3.workflows.test.ops_creator import ops_creator

DISPATCHER_ADDRESS = None # ("127.0.0.1", 10000)


class SingleTrainer:
    def __init__(self, name, ops_creator) -> None:
        self.name = name
        ops_name = [name for name in ops_creator if name.startswith(f"{self.name}.")].pop()
        if DISPATCHER_ADDRESS:
            self._ops = RemoteObj(ops_name, DISPATCHER_ADDRESS)
        else:
            self._ops = CoroutineAdapter(ops_creator[ops_name](ops_name))
        self._data = None

    def load_data(self, data):
        self._data = data

    async def train_step(self):
        await asyncio.gather(self._ops.step(self._data[0], self._data[1]))


class MultiTrainer:
    def __init__(self, name, ops_creator) -> None:
        self.name = name
        ops_names = [name for name in ops_creator if name.startswith(f"{self.name}.")]
        if DISPATCHER_ADDRESS:
            self._ops_list = [RemoteObj(ops_name, DISPATCHER_ADDRESS) for ops_name in ops_names]
        else:
            self._ops_list = [CoroutineAdapter(ops_creator[ops_name](ops_name)) for ops_name in ops_names]
        self._data = None

    def load_data(self, data):
        self._data = data

    async def train_step(self):
        for _ in range(3):
            await asyncio.gather(*[ops.step(self._data[0], self._data[1]) for ops in self._ops_list])


single_trainer = SingleTrainer("single", ops_creator)
single_trainer.load_data((torch.rand(10, 5), torch.randint(0, 2, (10,))))
multi_trainer = MultiTrainer("multi", ops_creator)
multi_trainer.load_data((torch.rand(15, 7), torch.randint(0, 3, (15,))))

batch_trainer = BatchTrainer([single_trainer, multi_trainer])

t0 = time.time()
batch_trainer.train()
print(f"total_time: {time.time() - t0}")
