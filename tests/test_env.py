# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import unittest

from dummy.dummy_business_engine import DummyEngine
from maro.simulator.core import BusinessEngineNotFoundError, Env


def run_to_end(env: Env):
    """Run the end of env"""
    is_done = False

    while not is_done:
        _, _, is_done = env.step(None)


class TestEnv(unittest.TestCase):
    """
    this test will use dummy scenario
    """        

    def test_builtin_scenario_with_default_parameters(self):
        """Test if the env with built-in scenario initializing correct"""
        max_tick = 10

        env = Env(scenario="cim", topology="toy.5p_ssddd_l0.0", durations=max_tick)

        run_to_end(env)

        # check port number
        ports_number = len(env.snapshot_list["ports"])

        self.assertEqual(ports_number, 5, msg=f"5pssddd topology should contains 5 ports, got {ports_number}")

    def test_env_interfaces_with_specified_business_engine_cls(self):
        """Test if env interfaces works as expect"""
        max_tick = 5

        env = Env(business_engine_cls=DummyEngine, start_tick=0, durations=max_tick)

        run_to_end(env)

        # check if the snapshot number equals with max_tick
        # NOTE: the snapshot_resolution defaults to 1, so the number of snapshots is same with max_tick
        num_of_snapshots = len(env.snapshot_list)

        self.assertEqual(max_tick, len(env.snapshot_list), msg=f"number of snapshots ({num_of_snapshots}) should be same "
                                                         f"with max tick ({max_tick}) without specified snapshot_resolution and max_snapshots")

        # check if we can reach to the end [start_tick, max_tick)
        self.assertEqual(max_tick-1, env.tick)

        # check if frame_index
        # NOTE: since we have not specified snapshot_resolution, frame_index should same with tick
        self.assertEqual(env.tick, env.frame_index)

        # check if config is same as we defined
        self.assertDictEqual(env.configs, {"name":"dummy"}, msg="configs should same as defined")

        # check node information
        node_info = env.summary["node_detail"]

        # check node exist
        self.assertTrue("dummies" in node_info, msg="dummy engine should contains dummy node")

        # check node number
        dummy_number = node_info["dummies"]["number"]

        self.assertEqual(10, dummy_number, msg=f"dummy should contains 10 nodes, got {dummy_number}")
        
        attributes = node_info["dummies"]["attributes"]

        # it will contains one attribute
        self.assertEqual(1, len(attributes), msg=f"dummy node should only contains 1 attribute, got {len(attributes)}")

        # and the attribute name is val
        self.assertTrue("val" in attributes)

        # attribute type should be i
        val_dtype = attributes['val']["type"]

        self.assertEqual("i", val_dtype, msg=f"dummy's val attribute should be int type, got {val_dtype}")

        # val should have only one slot (default)
        val_slots = attributes['val']["slots"]

        self.assertEqual(1, val_slots, msg=f"dummy's val attribute should be int type, got {val_slots}")

        # agent list should be [0, dummy_number)
        self.assertListEqual(list(range(0, dummy_number)), env.agent_idx_list, msg=f"dummy engine should have {dummy_number} agents")


        # check if snapshot list available
        self.assertIsNotNone(env.snapshot_list, msg="snapshot list should  be None")

        # reset should work
        
        dummies_ss = env.snapshot_list["dummies"]
        vals_before_reset = dummies_ss[env.frame_index::"val"]

        # before reset, snapshot should have value
        self.assertListEqual(list(vals_before_reset), [env.tick]*dummy_number, msg=f"we should have val value same as last tick, got {vals_before_reset}")

        env.reset()

        # after reset, it should 0
        vals_after_reset = dummies_ss[env.frame_index::"val"]

        self.assertListEqual(list(vals_after_reset), [0]*dummy_number, msg=f"we should have val value same as last tick, got {vals_after_reset}")

    def test_snapshot_resolution(self):
        """Test env with snapshot_resolution, it should take snapshot every snapshot_resolution ticks"""
        max_tick = 10

        env = Env(business_engine_cls=DummyEngine, start_tick=0, durations=max_tick, snapshot_resolution=3)

        run_to_end(env)

        # we should have 4 snapshots totally without max_snapshots speified
        self.assertEqual(4, len(env.snapshot_list), msg="We should have 4 snapshots in memory")

        # snapshot at 2, 5, 8, 9 ticks
        states = env.snapshot_list["dummies"][::"val"].reshape(-1, 10)
        
        # NOTE: frame_index is the index of frame in snapshot list, it is 0 based, so snapshot resolution will make tick not equals to frame_index
        # 
        for frame_index, tick in enumerate((2, 5, 8, 9)):
            self.assertListEqual(list(states[frame_index]), [tick] * 10, msg=f"states should be {tick}")


    def test_max_snapshots(self):
        """Test env  with max_snapshots, it should take snapshot every tick, but should last N kept"""
        max_tick = 10

        env = Env(business_engine_cls=DummyEngine, start_tick=0, durations=max_tick, max_snapshots=2)

        run_to_end(env)  

        # we should have 2 snapshots totally with max_snapshots speified
        self.assertEqual(2, len(env.snapshot_list), msg="We should have 2 snapshots in memory")

        # and only 87 and 9 in snapshot
        states = env.snapshot_list["dummies"][::"val"].reshape(-1, 10)
        
        # 1st should states at tick 7
        self.assertListEqual(list(states[0]), [8] * 10, msg="1st snapshot should be at tick 8")

        # 2nd should states at tick 9
        self.assertListEqual(list(states[1]), [9] * 10, msg="2nd snapshot should be at tick 9")

    def test_snapshot_resolution_with_max_snapshots(self):
        """Test env with both snapshot_resolution and max_snapshots parameters, and it should work as expected"""
        max_tick = 10

        env = Env(business_engine_cls=DummyEngine, start_tick=0, durations=max_tick, snapshot_resolution=2, max_snapshots=2)

        run_to_end(env)

        # we should have snapshot same as max_snapshots
        self.assertEqual(2, len(env.snapshot_list), msg="We should have 2 snapshots in memory")

        # and only 7 and 9 in snapshot
        states = env.snapshot_list["dummies"][::"val"].reshape(-1, 10)
        
        # 1st should states at tick 7
        self.assertListEqual(list(states[0]), [7] * 10, msg="1st snapshot should be at tick 7")

        # 2nd should states at tick 9
        self.assertListEqual(list(states[1]), [9] * 10, msg="2nd snapshot should be at tick 9")

    def test_early_stop(self):
        """Test if we can stop at specified tick with early stop at post_step function"""
        max_tick = 10

        env = Env(business_engine_cls=DummyEngine, start_tick=0, durations=max_tick,
            options={"post_step_early_stop": 6}) # early stop at tick 6, NOTE: simulator still

        run_to_end(env)

        # the end tick of env should be 6 as specified
        self.assertEqual(6, env.tick, msg=f"env should stop at tick 6, but {env.tick}")

        # avaiable snapshot should be 7 (0-6)
        states = env.snapshot_list["dummies"][::"val"].reshape(-1, 10)
        
        self.assertEqual(7, len(states), msg=f"available snapshot number should be 7, but {len(states)}")

        # and last one should be 6
        self.assertListEqual(list(states[-1]), [6]*10, msg="last states should be 6")

    def test_builtin_scenario_with_customized_topology(self):
        """Test using built-in scenario with customized topology"""

        max_tick = 10

        env = Env(scenario="cim", topology="tests/data/cim/customized_config", start_tick=0, durations=max_tick)

        run_to_end(env)

        # check if the config same as ours
        self.assertEqual([2], env.configs["container_volumes"], msg="customized container_volumes should be 2")

    def test_invalid_scenario(self):
        """Test specified invalid scenario"""

        # not exist scenario
        with self.assertRaises(ModuleNotFoundError) as ctx:
            env = Env("None", "toy.5p_ssddd_l0.0", 100)

        # not exist topology
        with self.assertRaises(FileNotFoundError) as ctx:
            env = Env("cim", "None", 100)



if __name__ == "__main__":
    unittest.main()
