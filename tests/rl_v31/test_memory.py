# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from tianshou.data import Batch
from maro.rl_v31.training.replay_memory import ReplayMemory, ReplayMemoryManager


class Dummy:
    def __init__(self, v: int) -> None:
        self.v = v

    def __repr__(self):
        return str(self.v)


def _get_batch(s: int, e: int) -> Batch:
    return Batch(
        obs=[Dummy(i) for i in range(s, e)],
        action=[Dummy(i) for i in range(s, e)],
        reward=[i % 2 for i in range(s, e)],
        next_obs=[Dummy(i) for i in range(s, e)],
        terminal=[i % 2 == 0 for i in range(s, e)],
        truncated=[i % 2 == 0 for i in range(s, e)],
    )


rmm = ReplayMemoryManager(
    memories=[ReplayMemory(capacity=10) for _ in range(3)]
)


rmm.store(batches=[_get_batch(0, 3), _get_batch(0, 4), _get_batch(0, 5)], ids=[0, 1, 2])
rmm.store(batches=[_get_batch(3, 6), _get_batch(4, 8), _get_batch(5, 10)], ids=[0, 1, 2])
rmm.store(batches=[_get_batch(6, 9), _get_batch(8, 12), _get_batch(10, 15)], ids=[0, 1, 2])

for memory in rmm.memories:
    print(memory.size, memory.ptr)
    print(memory.data)
print("\n" * 3)


t = rmm.sample(size=None, ids=None, random=False, pop=False)
for k, v in t.items():
    print(k, len(v), v)
print("\n" * 3)


t = rmm.sample(size=10, ids=None, random=True, pop=False)
for k, v in t.items():
    print(k, len(v), v)
print("\n" * 3)


t = rmm.sample(size=10, ids=None, random=False, pop=True)
for k, v in t.items():
    print(k, len(v), v)
for memory in rmm.memories:
    print(memory.size, memory.ptr)
    print(memory.data)
print("\n" * 3)


t = rmm.sample(size=None, ids=None, random=False, pop=False)
for k, v in t.items():
    print(k, len(v), v)
print("\n" * 3)
