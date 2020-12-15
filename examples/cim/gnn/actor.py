import ctypes
import multiprocessing
import os
import pickle
import time
from collections import OrderedDict
from multiprocessing import Pipe, Process

import numpy as np
import torch

from maro.rl import AbsActor
from maro.simulator import Env
from maro.simulator.scenarios.cim.common import Action

from .action_shaper import DiscreteActionShaper
from .experience_shaper import ExperienceShaper
from .shared_structure import SharedStructure
from .state_shaper import GNNStateShaper
from .utils import fix_seed, gnn_union


def organize_exp_list(experience_collections: dict, idx_mapping: dict):
    """The function assemble the experience from multiple processes into a dictionary.

    Args:
         experience_collections (dict): It stores the experience in all agents. The structure is the same as what is
            defined in the SharedStructure in the ParallelActor except additional key for experience length. For
            example:

            {
                "len": numpy.array,
                "s": {
                    "v": numpy.array,
                    "p": numpy.array,
                }
                "a": numpy.array,
                "R": numpy.array,
                "s_": {
                    "v": numpy.array,
                    "p": numpy.array,
                }
            }

            Note that the experience from different agents are stored in the same batch in a sequential way. For
            example, if agent x starts at b_x in batch index and the experience is l_x length long, the range [b_x,
            l_x) in the batch is the experience of agent x.

         idx_mapping (dict): The key is the name of each agent and the value is the starting index, e.g., b_x, of the
            storage space where the experience of the agent is stored.
    """
    result = {}
    tmpi = 0
    for code, idx in idx_mapping.items():
        exp_len = experience_collections["len"][0][tmpi]

        s = organize_obs(experience_collections["s"], idx, exp_len)
        s_ = organize_obs(experience_collections["s_"], idx, exp_len)
        R = experience_collections["R"][idx: idx + exp_len]
        R = R.reshape(-1, *R.shape[2:])
        a = experience_collections["a"][idx: idx + exp_len]
        a = a.reshape(-1, *a.shape[2:])

        result[code] = {
            "R": R,
            "a": a,
            "s": s,
            "s_": s_,
            "len": a.shape[0]
        }
        tmpi += 1
    return result


def organize_obs(obs, idx, exp_len):
    """Helper function to transform the observation from multiple processes to a unified dictionary."""
    tick_buffer, _, para_cnt, v_cnt, v_dim = obs["v"].shape
    _, _, _, p_cnt, p_dim = obs["p"].shape
    batch = exp_len * para_cnt
    # v: tick_buffer, seq_len,  parallel_cnt, v_cnt, v_dim --> (tick_buffer, cnt, v_cnt, v_dim)
    v = obs["v"][:, idx: idx + exp_len]
    v = v.reshape(tick_buffer, batch, v_cnt, v_dim)
    p = obs["p"][:, idx: idx + exp_len]
    p = p.reshape(tick_buffer, batch, p_cnt, p_dim)
    # vo: seq_len * parallel_cnt * v_cnt * p_cnt* --> cnt * v_cnt * p_cnt*
    vo = obs["vo"][idx: idx + exp_len]
    vo = vo.reshape(batch, v_cnt, vo.shape[-1])
    po = obs["po"][idx: idx + exp_len]
    po = po.reshape(batch, p_cnt, po.shape[-1])
    vedge = obs["vedge"][idx: idx + exp_len]
    vedge = vedge.reshape(batch, v_cnt, vedge.shape[-2], vedge.shape[-1])
    pedge = obs["pedge"][idx: idx + exp_len]
    pedge = pedge.reshape(batch, p_cnt, pedge.shape[-2], pedge.shape[-1])
    ppedge = obs["ppedge"][idx: idx + exp_len]
    ppedge = ppedge.reshape(batch, p_cnt, ppedge.shape[-2], ppedge.shape[-1])

    # mask: (seq_len, parallel_cnt, tick_buffer)
    mask = obs["mask"][idx: idx + exp_len].reshape(batch, tick_buffer)

    return {"v": v, "p": p, "vo": vo, "po": po, "pedge": pedge, "vedge": vedge, "ppedge": ppedge, "mask": mask}


def single_player_worker(index, config, exp_idx_mapping, pipe, action_io, exp_output):
    """The A2C worker function to collect experience.

    Args:
        index (int): The process index counted from 0.
        config (dict): It is a dottable dictionary that stores the configuration of the simulation, state_shaper and
            postprocessing shaper.
        exp_idx_mapping (dict): The key is agent code and the value is the starting index where the experience is stored
            in the experience batch.
        pipe (Pipe): The pipe instance for communication with the main process.
        action_io (SharedStructure): The shared memory to hold the state information that the main process uses to
            generate an action.
        exp_output (SharedStructure): The shared memory to transfer the experience list to the main process.
    """
    env = Env(**config.env.param)
    fix_seed(env, config.env.seed)
    static_code_list, dynamic_code_list = list(env.summary["node_mapping"]["ports"].values()), \
        list(env.summary["node_mapping"]["vessels"].values())
    # Create gnn_state_shaper without consuming any resources.

    gnn_state_shaper = GNNStateShaper(
        static_code_list, dynamic_code_list, config.env.param.durations, config.model.feature,
        tick_buffer=config.model.tick_buffer, max_value=env.configs["total_containers"])
    gnn_state_shaper.compute_static_graph_structure(env)

    action_io_np = action_io.structuralize()

    action_shaper = DiscreteActionShaper(config.model.action_dim)
    exp_shaper = ExperienceShaper(
        static_code_list, dynamic_code_list, config.env.param.durations, gnn_state_shaper,
        scale_factor=config.env.return_scaler, time_slot=config.training.td_steps,
        discount_factor=config.training.gamma, idx=index, shared_storage=exp_output.structuralize(),
        exp_idx_mapping=exp_idx_mapping)

    i = 0
    while pipe.recv() == "reset":
        env.reset()
        r, decision_event, is_done = env.step(None)

        j = 0
        logs = []
        while not is_done:
            model_input = gnn_state_shaper(decision_event, env.snapshot_list)
            action_io_np["v"][:, index] = model_input["v"]
            action_io_np["p"][:, index] = model_input["p"]
            action_io_np["vo"][index] = model_input["vo"]
            action_io_np["po"][index] = model_input["po"]
            action_io_np["vedge"][index] = model_input["vedge"]
            action_io_np["pedge"][index] = model_input["pedge"]
            action_io_np["ppedge"][index] = model_input["ppedge"]
            action_io_np["mask"][index] = model_input["mask"]
            action_io_np["pid"][index] = decision_event.port_idx
            action_io_np["vid"][index] = decision_event.vessel_idx
            pipe.send("features")
            model_action = pipe.recv()
            env_action = action_shaper(decision_event, model_action)
            exp_shaper.record(decision_event=decision_event, model_action=model_action, model_input=model_input)
            logs.append([
                index, decision_event.tick, decision_event.port_idx, decision_event.vessel_idx, model_action,
                env_action, decision_event.action_scope.load, decision_event.action_scope.discharge])
            action = Action(decision_event.vessel_idx, decision_event.port_idx, env_action)
            r, decision_event, is_done = env.step(action)
            j += 1
        action_io_np["sh"][index] = compute_shortage(env.snapshot_list, config.env.param.durations, static_code_list)
        i += 1
        pipe.send("done")
        gnn_state_shaper.end_ep_callback(env.snapshot_list)
        # Organize and synchronize exp to shared memory.
        exp_shaper(env.snapshot_list)
        exp_shaper.reset()
        logs = np.array(logs, dtype=np.float)
        pipe.send(logs)


def compute_shortage(snapshot_list, max_tick, static_code_list):
    """Helper function to compute the shortage after a episode end."""
    return np.sum(snapshot_list["ports"][max_tick - 1: static_code_list: "acc_shortage"])


class ParallelActor(AbsActor):
    def __init__(self, config, demo_env, gnn_state_shaper, agent_manager, logger):
        """A2C rollout class.

        This implements the synchronized A2C structure. Multiple processes are created to simulate and collect
        experience where only CPU is needed and whenever an action is required, they notify the main process and the
        main process will do the batch action inference with GPU.

        Args:
            config (dict): The configuration to run the simulation.
            demo_env (maro.simulator.Env): To get configuration information such as the amount of vessels and ports as
                well as the topology of the environment, the example environment is needed.
            gnn_state_shaper (AbsShaper): The state shaper instance to extract graph information from the state of
                the environment.
            agent_manager (AbsAgentManger): The agent manager instance to do the action inference in batch.
            logger: The logger instance to log information during the rollout.

        """
        super().__init__(demo_env, agent_manager)
        multiprocessing.set_start_method("spawn", True)
        self._logger = logger
        self.config = config

        self._static_node_mapping = demo_env.summary["node_mapping"]["ports"]
        self._dynamic_node_mapping = demo_env.summary["node_mapping"]["vessels"]
        self._gnn_state_shaper = gnn_state_shaper
        self.device = torch.device(config.training.device)

        self.parallel_cnt = config.training.parallel_cnt
        self.log_header = [f"sh_{i}" for i in range(self.parallel_cnt)]

        tick_buffer = config.model.tick_buffer

        v_dim, vedge_dim, v_cnt = self._gnn_state_shaper.get_input_dim("v"), \
            self._gnn_state_shaper.get_input_dim("vedge"), len(self._dynamic_node_mapping)
        p_dim, pedge_dim, p_cnt = self._gnn_state_shaper.get_input_dim("p"), \
            self._gnn_state_shaper.get_input_dim("pedge"), len(self._static_node_mapping)

        self.pipes = [Pipe() for i in range(self.parallel_cnt)]

        action_io_structure = {
            "p": ((tick_buffer, self.parallel_cnt, p_cnt, p_dim), ctypes.c_float),
            "v": ((tick_buffer, self.parallel_cnt, v_cnt, v_dim), ctypes.c_float),
            "po": ((self.parallel_cnt, p_cnt, v_cnt), ctypes.c_long),
            "vo": ((self.parallel_cnt, v_cnt, p_cnt), ctypes.c_long),
            "vedge": ((self.parallel_cnt, v_cnt, p_cnt, vedge_dim), ctypes.c_float),
            "pedge": ((self.parallel_cnt, p_cnt, v_cnt, vedge_dim), ctypes.c_float),
            "ppedge": ((self.parallel_cnt, p_cnt, p_cnt, pedge_dim), ctypes.c_float),
            "mask": ((self.parallel_cnt, tick_buffer), ctypes.c_bool),
            "sh": ((self.parallel_cnt, ), ctypes.c_long),
            "pid": ((self.parallel_cnt, ), ctypes.c_long),
            "vid": ((self.parallel_cnt, ), ctypes.c_long)
        }
        self.action_io = SharedStructure(action_io_structure)
        self.action_io_np = self.action_io.structuralize()

        tot_exp_len = sum(config.env.exp_per_ep.values())

        exp_output_structure = {
            "s": {
                "v": ((tick_buffer, tot_exp_len, self.parallel_cnt, v_cnt, v_dim), ctypes.c_float),
                "p": ((tick_buffer, tot_exp_len, self.parallel_cnt, p_cnt, p_dim), ctypes.c_float),
                "vo": ((tot_exp_len, self.parallel_cnt, v_cnt, p_cnt), ctypes.c_long),
                "po": ((tot_exp_len, self.parallel_cnt, p_cnt, v_cnt), ctypes.c_long),
                "vedge": ((tot_exp_len, self.parallel_cnt, v_cnt, p_cnt, vedge_dim), ctypes.c_float),
                "pedge": ((tot_exp_len, self.parallel_cnt, p_cnt, v_cnt, vedge_dim), ctypes.c_float),
                "ppedge": ((tot_exp_len, self.parallel_cnt, p_cnt, p_cnt, pedge_dim), ctypes.c_float),
                "mask": ((tot_exp_len, self.parallel_cnt, tick_buffer), ctypes.c_bool)
            },
            "s_": {
                "v": ((tick_buffer, tot_exp_len, self.parallel_cnt, v_cnt, v_dim), ctypes.c_float),
                "p": ((tick_buffer, tot_exp_len, self.parallel_cnt, p_cnt, p_dim), ctypes.c_float),
                "vo": ((tot_exp_len, self.parallel_cnt, v_cnt, p_cnt), ctypes.c_long),
                "po": ((tot_exp_len, self.parallel_cnt, p_cnt, v_cnt), ctypes.c_long),
                "vedge": ((tot_exp_len, self.parallel_cnt, v_cnt, p_cnt, vedge_dim), ctypes.c_float),
                "pedge": ((tot_exp_len, self.parallel_cnt, p_cnt, v_cnt, vedge_dim), ctypes.c_float),
                "ppedge": ((tot_exp_len, self.parallel_cnt, p_cnt, p_cnt, pedge_dim), ctypes.c_float),
                "mask": ((tot_exp_len, self.parallel_cnt, tick_buffer), ctypes.c_bool)
            },
            "a": ((tot_exp_len, self.parallel_cnt), ctypes.c_long),
            "len": ((self.parallel_cnt, len(config.env.exp_per_ep)), ctypes.c_long),
            "R": ((tot_exp_len, self.parallel_cnt, p_cnt), ctypes.c_float),
        }
        self.exp_output = SharedStructure(exp_output_structure)
        self.exp_output_np = self.exp_output.structuralize()

        self._logger.info("allocate complete")

        self.exp_idx_mapping = OrderedDict()
        acc_c = 0
        for key, c in config.env.exp_per_ep.items():
            self.exp_idx_mapping[key] = acc_c
            acc_c += c

        self.workers = [
            Process(
                target=single_player_worker,
                args=(i, config, self.exp_idx_mapping, self.pipes[i][1], self.action_io, self.exp_output)
            ) for i in range(self.parallel_cnt)
        ]
        for w in self.workers:
            w.start()

        self._logger.info("all thread started")

        self._roll_out_time = 0
        self._trainsfer_time = 0
        self._roll_out_cnt = 0

    def roll_out(self):
        """Rollout using current policy in the AgentManager.

        Returns:
            result (dict): The key is the agent code, the value is the experience list stored in numpy.array.
        """
        # Compute the time used for state preparation in the child process.
        t_state = 0
        # Compute the time used for action inference.
        t_action = 0

        for p in self.pipes:
            p[0].send("reset")
        self._roll_out_cnt += 1

        step_i = 0
        tick = time.time()
        while True:
            signals = [p[0].recv() for p in self.pipes]
            if signals[0] == "done":
                break

            step_i += 1

            t = time.time()
            graph = gnn_union(
                self.action_io_np["p"], self.action_io_np["po"], self.action_io_np["pedge"],
                self.action_io_np["v"], self.action_io_np["vo"], self.action_io_np["vedge"],
                self._gnn_state_shaper.p2p_static_graph, self.action_io_np["ppedge"],
                self.action_io_np["mask"], self.device
            )
            t_state += time.time() - t

            assert(np.min(self.action_io_np["pid"]) == np.max(self.action_io_np["pid"]))
            assert(np.min(self.action_io_np["vid"]) == np.max(self.action_io_np["vid"]))

            t = time.time()
            actions = self._inference_agents.choose_action(
                agent_id=(self.action_io_np["pid"][0], self.action_io_np["vid"][0]), state=graph
            )
            t_action += time.time() - t

            for i, p in enumerate(self.pipes):
                p[0].send(actions[i])

        self._roll_out_time += time.time() - tick
        tick = time.time()
        self._logger.info("receiving exp")
        logs = [p[0].recv() for p in self.pipes]

        self._logger.info(f"Mean of shortage: {np.mean(self.action_io_np['sh'])}")
        self._trainsfer_time += time.time() - tick

        self._logger.debug(dict(zip(self.log_header, self.action_io_np["sh"])))

        with open(os.path.join(self.config.log.path, f"logs_{self._roll_out_cnt}"), "wb") as fp:
            pickle.dump(logs, fp)

        self._logger.info("organize exp_dict")
        result = organize_exp_list(self.exp_output_np, self.exp_idx_mapping)

        if self.config.log.exp.enable and self._roll_out_cnt % self.config.log.exp.freq == 0:
            with open(os.path.join(self.config.log.path, f"exp_{self._roll_out_cnt}"), "wb") as fp:
                pickle.dump(result, fp)

        self._logger.debug(f"play time: {int(self._roll_out_time)}")
        self._logger.debug(f"transfer time: {int(self._trainsfer_time)}")
        return result

    def exit(self):
        """Terminate the child processes."""
        for p in self.pipes:
            p[0].send("close")
