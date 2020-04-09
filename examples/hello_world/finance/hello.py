from maro.simulator import Env
from maro.simulator.frame import SnapshotList


env = Env("finance", "test", max_tick=10)

reward, decision_event, is_done = env.step(None)

while not is_done:
    reward, decision_event, is_done = env.step(None)

stock_snapshots : SnapshotList = env.snapshot_list.test_stocks

print("len of snapshot:", len(stock_snapshots))

stock_opening_price = stock_snapshots.static_nodes[::("opening_price", 0)]

print("opening price for all the ticks:")
print(stock_opening_price)