import random
import time
from functools import reduce

pool = range(100)
n = 100000

l = [random.choices(pool, k=10) for _ in range(n)]

t0 = time.time()
for vals in l:
    reduce(lambda x, y: x + y, vals)
t1 = time.time()
[reduce(lambda x, y: x + y, vals) for vals in l]
t2 = time.time()

print(t1 - t0, t2 - t1)
