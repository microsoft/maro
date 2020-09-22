from maro.rl import TwoPhaseLinearExplorer

exploration_config = {"epsilon_range_dict": {"_all_": (.0, .4)},
                      "split_point_dict": {"_all_": (.5, .8)},
                      "with_cache": True
                      }
