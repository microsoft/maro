Multi Agent DQN for CIM
================================================

This example demonstrates how to use MARO's reinforcement learning (RL) toolkit to solve the container
inventory management (CIM) problem. It is formalized as a multi-agent reinforcement learning problem,
where each port acts as a decision agent. When a vessel arrives at a port, these agents must take actions
by transferring a certain amount of containers to / from the vessel. The objective is for the agents to
learn policies that minimize the overall container shortage.

Trajectory
----------

The ``CIMTrajectoryForDQN`` inherits from ``Trajectory`` function and implements methods to be used as callbacks
in the roll-out loop. In this example,
  * ``get_state`` converts environment observations to state vectors that encode temporal and spatial information.
    The temporal information includes relevant port and vessel information, such as shortage and remaining space,
    over the past k days (here k = 7). The spatial information includes features of the downstream ports.
  * ``get_action`` converts agents' output (an integer that maps to a percentage of containers to be loaded
    to or unloaded from the vessel) to action objects that can be executed by the environment.
  * ``get_offline_reward`` computes the reward of a given action as a linear combination of fulfillment and
    shortage within a future time frame.
  * ``on_finish`` processes a complete trajectory into data that can be used directly by the learning agents.


.. code-block:: python
    class CIMTrajectoryForDQN(Trajectory):
        def __init__(
            self, env, *, port_attributes, vessel_attributes, action_space, look_back, max_ports_downstream,
            reward_time_window, fulfillment_factor, shortage_factor, time_decay,
            finite_vessel_space=True, has_early_discharge=True
        ):
            super().__init__(env)
            self.port_attributes = port_attributes
            self.vessel_attributes = vessel_attributes
            self.action_space = action_space
            self.look_back = look_back
            self.max_ports_downstream = max_ports_downstream
            self.reward_time_window = reward_time_window
            self.fulfillment_factor = fulfillment_factor
            self.shortage_factor = shortage_factor
            self.time_decay = time_decay
            self.finite_vessel_space = finite_vessel_space
            self.has_early_discharge = has_early_discharge

        def get_state(self, event):
            vessel_snapshots, port_snapshots = self.env.snapshot_list["vessels"], self.env.snapshot_list["ports"]
            tick, port_idx, vessel_idx = event.tick, event.port_idx, event.vessel_idx
            ticks = [max(0, tick - rt) for rt in range(self.look_back - 1)]
            future_port_idx_list = vessel_snapshots[tick: vessel_idx: 'future_stop_list'].astype('int')
            port_features = port_snapshots[ticks: [port_idx] + list(future_port_idx_list): self.port_attributes]
            vessel_features = vessel_snapshots[tick: vessel_idx: self.vessel_attributes]
            return {port_idx: np.concatenate((port_features, vessel_features))}

        def get_action(self, action_by_agent, event):
            vessel_snapshots = self.env.snapshot_list["vessels"]
            action_info = list(action_by_agent.values())[0]
            model_action = action_info[0] if isinstance(action_info, tuple) else action_info
            scope, tick, port, vessel = event.action_scope, event.tick, event.port_idx, event.vessel_idx
            zero_action_idx = len(self.action_space) / 2  # index corresponding to value zero.
            vessel_space = vessel_snapshots[tick:vessel:self.vessel_attributes][2] if self.finite_vessel_space else float("inf")
            early_discharge = vessel_snapshots[tick:vessel:"early_discharge"][0] if self.has_early_discharge else 0
            percent = abs(self.action_space[model_action])

            if model_action < zero_action_idx:
                action_type = ActionType.LOAD
                actual_action = min(round(percent * scope.load), vessel_space)
            elif model_action > zero_action_idx:
                action_type = ActionType.DISCHARGE
                plan_action = percent * (scope.discharge + early_discharge) - early_discharge
                actual_action = round(plan_action) if plan_action > 0 else round(percent * scope.discharge)
            else:
                actual_action, action_type = 0, ActionType.LOAD

            return {port: Action(vessel, port, actual_action, action_type)}

        def get_offline_reward(self, event):
            port_snapshots = self.env.snapshot_list["ports"]
            start_tick = event.tick + 1
            ticks = list(range(start_tick, start_tick + self.reward_time_window))

            future_fulfillment = port_snapshots[ticks::"fulfillment"]
            future_shortage = port_snapshots[ticks::"shortage"]
            decay_list = [
                self.time_decay ** i for i in range(self.reward_time_window)
                for _ in range(future_fulfillment.shape[0] // self.reward_time_window)
            ]

            tot_fulfillment = np.dot(future_fulfillment, decay_list)
            tot_shortage = np.dot(future_shortage, decay_list)

            return np.float32(self.fulfillment_factor * tot_fulfillment - self.shortage_factor * tot_shortage)

        def on_env_feedback(self, event, state_by_agent, action_by_agent, reward):
            self.trajectory["event"].append(event)
            self.trajectory["state"].append(state_by_agent)
            self.trajectory["action"].append(action_by_agent)

        def on_finish(self):
            exp_by_agent = defaultdict(lambda: defaultdict(list))
            for i in range(len(self.trajectory["state"]) - 1):
                agent_id = list(self.trajectory["state"][i].keys())[0]
                exp = exp_by_agent[agent_id]
                exp["S"].append(self.trajectory["state"][i][agent_id])
                exp["A"].append(self.trajectory["action"][i][agent_id])
                exp["R"].append(self.get_offline_reward(self.trajectory["event"][i]))
                exp["S_"].append(list(self.trajectory["state"][i + 1].values())[0])

            return dict(exp_by_agent)


Agent
-----

The out-of-the-box DQN is used as our agent.

.. code-block:: python
    agent_config = {
        "model": ...,
        "optimization": ...,
        "hyper_params": ...
    }

    def get_dqn_agent():
        q_model = SimpleMultiHeadModel(
            FullyConnectedBlock(**agent_config["model"]), optim_option=agent_config["optimization"]
        )
        return DQN(q_model, DQNConfig(**agent_config["hyper_params"]))


Training
--------

The distributed training consists of one learner process and multiple actor processes. The learner optimizes
the policy by collecting roll-out data from the actors to train the underlying agents.

The actor process must create a roll-out executor for performing the requested roll-outs, which means that the
the environment simulator and shapers should be created here. In this example, inference is performed on the
actor's side, so a set of DQN agents must be created in order to load the models (and exploration parameters)
from the learner.

.. code-block:: python
    def cim_dqn_actor():
        env = Env(**training_config["env"])
        agent = MultiAgentWrapper({name: get_dqn_agent() for name in env.agent_idx_list})
        actor = Actor(env, agent, CIMTrajectoryForDQN, trajectory_kwargs=common_config)
        actor.as_worker(training_config["group"])

The learner's side requires a concrete learner class that inherits from ``AbsLearner`` and implements the ``run``
method which contains the main training loop. Here the implementation is similar to the single-threaded version
except that the ``collect`` method is used to obtain roll-out data from the actors (since the roll-out executors
are located on the actors' side). The agents created here are where training occurs and hence always contains the
latest policies.

.. code-block:: python
    def cim_dqn_learner():
        env = Env(**training_config["env"])
        agent = MultiAgentWrapper({name: get_dqn_agent() for name in env.agent_idx_list})
        scheduler = TwoPhaseLinearParameterScheduler(training_config["max_episode"], **training_config["exploration"])
        actor = ActorProxy(
            training_config["group"], training_config["num_actors"],
            update_trigger=training_config["learner_update_trigger"]
        )
        learner = OffPolicyLearner(actor, scheduler, agent, **training_config["training"])
        learner.run()

.. note::

  All related code snippets are supported in `maro playground <https://hub.docker.com/r/maro2020/playground>`_.
