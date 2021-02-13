Multi Agent DQN for CIM
================================================

This example demonstrates how to use MARO's reinforcement learning (RL) toolkit to solve the
`CIM <https://maro.readthedocs.io/en/latest/scenarios/container_inventory_management.html>`_ problem. It is formalized as a multi-agent reinforcement learning problem, where each port acts as a decision
agent. The agents take actions independently, e.g., loading containers to vessels or discharging containers from vessels.

State Shaper
------------

`State shaper <https://maro.readthedocs.io/en/latest/key_components/rl_toolkit.html#shapers>`_ converts the environment
observation to the model input state which includes temporal and spatial information. For this scenario, the model input
state includes:

- Temporal information, including the past week's information of ports and vessels, such as shortage on port and
remaining space on vessel.
- Spatial information, including related downstream port features.

.. code-block:: python
    PORT_ATTRIBUTES = ["empty", "full", "on_shipper", "on_consignee", "booking", "shortage", "fulfillment"]
    VESSEL_ATTRIBUTES = ["empty", "full", "remaining_space"]
    LOOK_BACK = 7
    MAX_PORTS_DOWNSTREAM = 2

    class CIMStateShaper(StateShaper):
        ...
        def __call__(self, decision_event, snapshot_list):
            tick, port_idx, vessel_idx = decision_event.tick, decision_event.port_idx, decision_event.vessel_idx
            ticks = [tick - rt for rt in range(self._look_back - 1)]
            future_port_idx_list = snapshot_list["vessels"][tick: vessel_idx: 'future_stop_list'].astype('int')
            port_features = snapshot_list["ports"][ticks: [port_idx] + list(future_port_idx_list): PORT_ATTRIBUTES]
            vessel_features = snapshot_list["vessels"][tick: vessel_idx: VESSEL_ATTRIBUTES]
            state = np.concatenate((port_features, vessel_features))
            return state


Action Shaper
-------------

`Action shaper <https://maro.readthedocs.io/en/latest/key_components/rl_toolkit.html#shapers>`_ is used to convert an
agent's model output to an environment executable action. For this specific scenario, the action space consists of
integers from -10 to 10, with -10 indicating loading 100% of the containers in the current inventory to the vessel and
10 indicating discharging 100% of the containers on the vessel to the port.

.. code-block:: python

    class CIMActionShaper(ActionShaper):
        ...
        def __call__(self, model_action, decision_event, snapshot_list):
            scope = decision_event.action_scope
            tick = decision_event.tick
            port_idx = decision_event.port_idx
            vessel_idx = decision_event.vessel_idx

            port_empty = snapshot_list["ports"][tick: port_idx: ["empty", "full", "on_shipper", "on_consignee"]][0]
            vessel_remaining_space = snapshot_list["vessels"][tick: vessel_idx: ["empty", "full", "remaining_space"]][2]
            early_discharge = snapshot_list["vessels"][tick:vessel_idx: "early_discharge"][0]
            assert 0 <= model_action < len(self._action_space)
            operation_num = self._action_space[model_action]

            if model_action < self._zero_action_index:
                actual_action = max(round(operation_num * port_empty), -vessel_remaining_space)
                action_type = ActionType.LOAD
            elif model_action > self._zero_action_index:
                plan_action = operation_num * (scope.discharge + early_discharge) - early_discharge
                actual_action = round(plan_action) if plan_action > 0 else round(operation_num * scope.discharge)
                action_type = ActionType.DISCHARGE
            else:
                actual_action = 0
                action_type = None

            return Action(vessel_idx, port_idx, abs(actual_action), action_type)

Experience Shaper
-----------------

`Experience shaper <https://maro.readthedocs.io/en/latest/key_components/rl_toolkit.html#shapers>`_ is used to convert
an episode trajectory to trainable experiences for RL agents. For this specific scenario, the reward is a linear
combination of fulfillment and shortage in a limited time window.

.. code-block:: python
    class CIMExperienceShaper(ExperienceShaper):
        ...
        def __call__(self, trajectory, snapshot_list):
            states = self._trajectory["state"]
            actions = self._trajectory["action"]
            agent_ids = self._trajectory["agent_id"]
            events = self._trajectory["event"]

            experiences_by_agent = defaultdict(lambda: defaultdict(list))
            for i in range(len(states) - 1):
                experiences = experiences_by_agent[agent_ids[i]]
                experiences["state"].append(states[i])
                experiences["action"].append(actions[i])
                experiences["reward"].append(self._compute_reward(events[i], snapshot_list))
                experiences["next_state"].append(states[i + 1])

            return dict(experiences_by_agent)

Agent
-----

`Agent <https://maro.readthedocs.io/en/latest/key_components/rl_toolkit.html#agent>`_ is the
kernel abstraction of the RL formulation for a real-world problem. For this scenario, the agent
is the algorithmic abstraction of a port. We choose DQN as our underlying learning algorithm
with a TD-error-based sampling mechanism.

.. code-block:: python    
    def create_dqn_agent():
        q_net = FullyConnectedBlock(
            input_dim=(LOOK_BACK + 1) * (MAX_PORTS_DOWNSTREAM + 1) * len(PORT_ATTRIBUTES) + len(VESSEL_ATTRIBUTES),
            hidden_dims=[256, 128, 64],
            output_dim=21,  # action space [0, 1, ..., 20]
            activation=nn.LeakyReLU,
            is_head=True,
            batch_norm=True, 
            softmax=False,
            skip_connection=False,
            dropout_p=.0
        )

        return DQN( 
            SimpleMultiHeadModel(q_net, optim_option=OptimOption(optim_cls=RMSprop, optim_params={"lr": 0.05})),
            DQNConfig(
                reward_discount=.0, 
                min_exp_to_train=1024,
                num_batches=10,
                batch_size=128, 
                target_update_freq=5, 
                tau=0.1, 
                is_double=True, 
                per_sample_td_error=True,
                loss_cls=nn.SmoothL1Loss
            )
        )

Roll-out Loop
-------------


.. code-block:: python
    class BasicRolloutExecutor(AbsRolloutExecutor):
        ...
        def roll_out(self, index, training=True):
            self.env.reset()
            metrics, event, is_done = self.env.step(None)
            while not is_done:
                state = self.state_shaper(event, self.env.snapshot_list)
                agent_id = event.port_idx
                action = self.agent[agent_id].choose_action(state)
                self.experience_shaper.record(
                    {"state": state, "agent_id": agent_id, "event": event, "action": action}
                )
                metrics, event, is_done = self.env.step(self.action_shaper(action, event, self.env.snapshot_list))

            exp = self.experience_shaper(self.env.snapshot_list) if training else None
            self.experience_shaper.reset()

            return exp


Single-threaded Training
------------------------

This single-threaded training workflow is comprised of the following steps:
- Initialize an environment with specific scenario and topology parameters. 
- Create agents and shapers.
- Execute the training loop with a scheduler and a roll-out executor. 

.. code-block::python
    env = Env("cim", "toy.4p_ssdd_l0.0", durations=1120)
    state_shaper = CIMStateShaper(look_back=LOOK_BACK, max_ports_downstream=MAX_PORTS_DOWNSTREAM)
    action_shaper = CIMActionShaper(action_space=list(np.linspace(-1.0, 1.0, 21)))
    experience_shaper = CIMExperienceShaper(time_window=100, fulfillment_factor=1.0, shortage_factor=1.0, time_decay_factor=0.97)
    agent = MultiAgentWrapper({name: create_dqn_agent() for name in env.agent_idx_list})
    scheduler = TwoPhaseLinearParameterScheduler(
        max_iter=100,
        parameter_names=["epsilon"],
        split_ep=50,
        start_values=0.4,
        mid_values=0.32,
        end_values=.0
    )
    executor = BasicRolloutExecutor(env, agent, state_shaper, action_shaper, experience_shaper)
    for exploration_params in scheduler:
        agent.set_exploration_params(exploration_params)
        exp_by_agent = executor.roll_out(scheduler.iter)
        print(f"ep {scheduler.iter} - metrics: {env.metrics}, exploration_params: {exploration_params}")
        for agent_id, exp in exp_by_agent.items():
            exp.update({"loss": [1e8] * len(list(exp.values())[0])})
            agent[agent_id].store_experiences(exp)

        for dqn in agent.agent_dict.values():
            dqn.train()


Distributed Training
--------------------

We demonstrate a single-learner and multi-actor topology where the learner drives the program by telling remote actors
to perform roll-out tasks and using the results they sent back to improve policies. The workflow usually involves
launching the learner process and actor processes separately.

On the actor side, the agent manager must be equipped with all shapers as well as an explorer. Thus, The code for
creating an environment and an agent manager on the actor side is similar to that for the single-host version. As in the
single-process version, the environment and the agent manager are wrapped in a SimpleActor instance. To make the actor a
distributed worker, we need to further wrap it in an ActorWorker instance. Finally, we launch the worker and it starts to
listen to roll-out requests from the learner. The following code snippet shows the creation of an actor worker with a
simple(local) actor wrapped inside.

.. code-block:: python
    env = Env("cim", "toy.4p_ssdd_l0.0", durations=1120)
    state_shaper = CIMStateShaper(look_back=LOOK_BACK, max_ports_downstream=MAX_PORTS_DOWNSTREAM)
    action_shaper = CIMActionShaper(action_space=list(np.linspace(-1.0, 1.0, 21)))
    experience_shaper = CIMExperienceShaper(time_window=100, fulfillment_factor=1.0, shortage_factor=1.0, time_decay_factor=0.97)

    agent = MultiAgentWrapper({name: create_dqn_agent() for name in env.agent_idx_list})
    executor = BasicRolloutExecutor(env, agent, state_shaper, action_shaper, experience_shaper)
    actor = BaseActor("cim-dqn", executor)
    actor.run()

On the learner side, instead of creating an actor, we create an actor proxy and wrap it inside the learner. This proxy
serves as the communication interface for the learner and is responsible for sending roll-out requests to remote actor
processes and receiving results. Calling the train method executes the usual training loop except that the actual
roll-out is performed remotely. The code snippet below shows the creation of a learner with an actor proxy wrapped
inside that communicates with 3 actors. 

.. code-block:: python
    env = Env("cim", "toy.4p_ssdd_l0.0", durations=1120)
    agent = MultiAgentWrapper({name: create_dqn_agent() for name in env.agent_idx_list})
    scheduler = TwoPhaseLinearParameterScheduler(
        max_iter=100,
        parameter_names=["epsilon"],
        split_ep=50,
        start_values=0.4,
        mid_values=0.32,
        end_values=.0
    )

    learner = SimpleLearner(""cim-dqn"", 3, agent, scheduler)
    learner.run()
    learner.exit()

.. note::

  All related code snippets are supported in `maro playground <https://hub.docker.com/r/arthursjiang/maro>`_.
