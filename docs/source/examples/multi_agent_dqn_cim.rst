Multi Agent DQN for CIM
================================================

This example demonstrates how to use MARO's reinforcement learning (RL) toolkit to solve the container
inventory management (CIM) problem. It is formalized as a multi-agent reinforcement learning problem,
where each port acts as a decision agent. When a vessel arrives at a port, these agents must take actions
by transfering a certain amount of containers to / from the vessel. The objective is for the agents to
learn policies that minimize the overall container shortage.

State Shaping
-------------

The function below converts environment observations to state vectors that encode temporal and spatial
information. The temporal information includes relevant port and vessel information, such as shortage
and remaining space, over the past k days (here k = 7). The spatial information includes features
of the downstream ports..

.. code-block:: python
    def get_state(decision_event, snapshots, look_back=7):
        tick, port_idx, vessel_idx = decision_event.tick, decision_event.port_idx, decision_event.vessel_idx
        ticks = [tick - rt for rt in range(look_back - 1)]
        future_port_idx_list = snapshots["vessels"][tick: vessel_idx: 'future_stop_list'].astype('int')
        port_features = snapshots["ports"][ticks: [port_idx] + list(future_port_idx_list): PORT_ATTRIBUTES]
        vessel_features = snapshots["vessels"][tick: vessel_idx: VESSEL_ATTRIBUTES]
        return np.concatenate((port_features, vessel_features))


Action Shaping
--------------

The function below converts agents' output (an integer that maps to a percentage of containers to be loaded
to or unloaded from the vessel) to action objects that can be executed by the environment.

.. code-block:: python
    def get_env_action(model_action, decision_event, vessel_snapshots):
        scope = decision_event.action_scope
        tick = decision_event.tick
        port = decision_event.port_idx
        vessel = decision_event.vessel_idx
        zero_action_idx = len(ACTION_SPACE) / 2  # index corresponding to value zero.

        vessel_space = vessel_snapshots[tick:vessel:VESSEL_ATTRIBUTES][2]
        early_discharge = vessel_snapshots[tick:vessel:"early_discharge"][0]
        percent = abs(ACTION_SPACE[model_action])
        if model_action < zero_action_idx:
            action_type = ActionType.LOAD
            actual_action = min(round(percent * scope.load), vessel_space)
        elif model_action > zero_action_idx:
            action_type = ActionType.DISCHARGE
            plan_action = percent * (scope.discharge + early_discharge) - early_discharge
            actual_action = round(plan_action) if plan_action > 0 else round(percent * scope.discharge)
        else:
            actual_action, action_type = 0, None

        return Action(vessel, port, actual_action, action_type)

Reward Shaping
--------------

The function below computes the reward of a given action as a linear combination of fulfillment and
shortage within a future time frame (set to 100 here).

.. code-block:: python
    def get_reward(
        decision_event, port_snapshots, reward_time_window=100, time_decay=0.97,
        fulfillment_factor=1.0, shortage_factor=1.0    
    ):
        start_tick = decision_event.tick + 1
        end_tick = decision_event.tick + reward_time_window
        ticks = list(range(start_tick, end_tick))
        future_fulfillment = port_snapshots[ticks::"fulfillment"]
        future_shortage = port_snapshots[ticks::"shortage"]
        decay_list = [
            time_decay ** i for i in range(end_tick - start_tick)
            for _ in range(future_fulfillment.shape[0] // (end_tick - start_tick))
        ]

        tot_fulfillment = np.dot(future_fulfillment, decay_list)
        tot_shortage = np.dot(future_shortage, decay_list)

        return np.float32(fulfillment_factor * tot_fulfillment - shortage_factor * tot_shortage)


Training Data Processing
------------------------

The function below processes a trajectory of transitions into training data. The transitions are
bucketed by the agent ID.

.. code-block:: python
    def get_training_data(trajectory, port_snapshots):
        states = trajectory["state"]
        actions = trajectory["action"]
        agent_ids = trajectory["agent_id"]
        events = trajectory["event"]

        exp_by_agent = defaultdict(lambda: defaultdict(list))
        for i in range(len(states) - 1):
            exp = exp_by_agent[agent_ids[i]]
            exp["state"].append(states[i])
            exp["action"].append(actions[i])
            exp["reward"].append(get_reward(events[i], port_snapshots))
            exp["next_state"].append(states[i + 1])

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

Roll-out Loop
-------------

The roll-out loop is highly customizable and its implementation is left to the user through the
``AbsRolloutExecutor`` interface. There is generally no restriction on the type of data the routine
should return, so long as the user knows what to do with it. But if the ``training`` option is set to
true. it is expected to return (or store in an externally accessible data structure) data needed for
model training. Below is an implementation of the roll-out loop. Note how the shaping functions are
used during the agents' interaction with the environment. For each transition, we record the agent ID,
event, state, action and its log probability. At the end of the roll-out, the recorded sequence of
transitions (the trajectory) gets processed into training data. 

.. code-block:: python
    class BasicRolloutExecutor(AbsRolloutExecutor):
        def roll_out(self, index, training=True, model_dict=None, exploration_params=None):
            self.env.reset()
            trajectory = {key: [] for key in ["state", "action", "agent_id", "event"]}
            if model_dict:
                self.agent.load_model(model_dict)  
            if exploration_params:
                self.agent.set_exploration_params(exploration_params)
            metrics, event, is_done = self.env.step(None)
            while not is_done:
                state = get_state(event, self.env.snapshot_list)
                agent_id = event.port_idx
                action = self.agent[agent_id].choose_action(state)
                trajectory["state"].append(state)
                trajectory["agent_id"].append(agent_id)
                trajectory["event"].append(event)
                trajectory["action"].append(action)
                env_action = get_env_action(action, event, self.env.snapshot_list["vessels"])
                metrics, event, is_done = self.env.step(env_action)

            return get_training_data(trajectory, self.env.snapshot_list["ports"]) if training else None


Distributed Training
--------------------

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
        executor = BasicRolloutExecutor(env, agent)
        actor = BaseActor(training_config["group"], executor)
        actor.run()

The learner's side requires a concrete learner class that inherits from ``AbsLearner`` and implements the ``run``
method which contains the main training loop. Here the implementation is similar to the single-threaded version
except that the ``collect`` method is used to obtain roll-out data from the actors (since the roll-out executors
are located on the actors' side). The agents created here are where training occurs and hence always contains the
latest policies. 

.. code-block:: python
    class BasicLearner(AbsLearner):
        ...
        def run(self):
            for exploration_params in self.scheduler:
                metrics_by_src, exp_by_src = self.collect(
                    self.scheduler.iter, model_dict=self.agent.dump_model(), exploration_params=exploration_params
                )
                for agent_id, exp in concat(exp_by_src).items():
                    exp.update({"loss": [1e8] * len(list(exp.values())[0])})
                    self.agent[agent_id].store_experiences(exp)

                for agent in self.agent.agent_dict.values():
                    agent.train()


    def cim_dqn_learner():
        env = Env(**training_config["env"])
        agent = MultiAgentWrapper({name: get_dqn_agent() for name in env.agent_idx_list})
        scheduler = TwoPhaseLinearParameterScheduler(training_config["max_episode"], **training_config["exploration"])
        learner = BasicLearner(
            training_config["group"], training_config["num_actors"], agent, scheduler,
            update_trigger=training_config["learner_update_trigger"]
        )

        time.sleep(5)
        learner.run()
        learner.exit()

.. note::

  All related code snippets are supported in `maro playground <https://hub.docker.com/r/arthursjiang/maro>`_.
