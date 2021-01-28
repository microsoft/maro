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

    class CIMStateShaper(StateShaper):
        ...
        def __call__(self, decision_event, snapshot_list):
            tick, port_idx, vessel_idx = decision_event.tick, decision_event.port_idx, decision_event.vessel_idx
            ticks = [tick - rt for rt in range(self._look_back - 1)]
            future_port_idx_list = snapshot_list["vessels"][tick: vessel_idx: 'future_stop_list'].astype('int')
            port_features = snapshot_list["ports"][ticks: [port_idx] + list(future_port_idx_list): PORT_ATTRIBUTES]
            vessel_features = snapshot_list["vessels"][tick: vessel_idx: VESSEL_ATTRIBUTES]
            state = np.concatenate((port_features, vessel_features))
            return str(port_idx), state


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

            if model_action < self._zero_action_index:
                actual_action = max(round(self._action_space[model_action] * port_empty), -vessel_remaining_space)
            elif model_action > self._zero_action_index:
                plan_action = self._action_space[model_action] * (scope.discharge + early_discharge) - early_discharge
                actual_action = (
                    round(plan_action) if plan_action > 0
                    else round(self._action_space[model_action] * scope.discharge)
                )
            else:
                actual_action = 0

            return Action(vessel_idx, port_idx, actual_action)

Experience Shaper
-----------------

`Experience shaper <https://maro.readthedocs.io/en/latest/key_components/rl_toolkit.html#shapers>`_ is used to convert
an episode trajectory to trainable experiences for RL agents. For this specific scenario, the reward is a linear
combination of fulfillment and shortage in a limited time window.

.. code-block:: python
    class TruncatedExperienceShaper(ExperienceShaper):
        ...
        def __call__(self, trajectory, snapshot_list):
            experiences_by_agent = defaultdict(lambda: defaultdict(list))
            states = trajectory["state"]
            actions = trajectory["action"]
            agent_ids = trajectory["agent_id"]
            events = trajectory["event"]
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
    NUM_ACTIONS = 21
    class DQNAgent(AbsAgent):
        ...
        def train(self):
            if len(self._experience_pool) < self._min_experiences_to_train:
                return

            for _ in range(self._num_batches):
                indexes, sample = self._experience_pool.sample_by_key("loss", self._batch_size)
                state = np.asarray(sample["state"])
                action = np.asarray(sample["action"])
                reward = np.asarray(sample["reward"])
                next_state = np.asarray(sample["next_state"])
                loss = self._algorithm.train(state, action, reward, next_state)
                self._experience_pool.update(indexes, {"loss": loss})

    def create_dqn_agents(agent_id_list):
        agent_dict = {}
        for agent_id in agent_id_list:
            q_net = NNStack(
                "q_value",
                FullyConnectedBlock(
                    input_dim=state_shaper.dim,
                    hidden_dims=[256, 128, 64],
                    output_dim=NUM_ACTIONS,
                    activation=nn.LeakyReLU,
                    is_head=True,
                    batch_norm_enabled=True, 
                    softmax_enabled=False,
                    skip_connection_enabled=False,
                    dropout_p=.0)
            )
            
            learning_model = SimpleMultiHeadModel(
                q_net, optimizer_options=OptimizerOptions(cls=RMSprop, params={"lr": 0.05})
            )
            agent_dict[agent_id] = DQN(
                agent_id, 
                learning_model, 
                config=DQNConfig(
                    reward_discount=.0, 
                    min_experiences_to_train=1024,
                    num_batches=10,
                    batch_size=128, 
                    target_update_frequency=5, 
                    tau=0.1, 
                    is_double=True, 
                    per_sample_td_error_enabled=True,
                    loss_cls=nn.SmoothL1Loss
                ),
                experience_pool=SimpleStore(["state", "action", "reward", "next_state", "loss"])
            )

        return agent_dict

Agent Manager
-------------

The complexities of the environment can be isolated from the learning algorithm by using an
`Agent manager <https://maro.readthedocs.io/en/latest/key_components/rl_toolkit.html#agent-manager>`_
to manage individual agents. We define a function to create the agents and an agent manager class
that implements the ``train`` method where the newly obtained experiences are stored in the agents'
experience pools before training, in accordance with the DQN algorithm.

.. code-block:: python
    class DQNAgentManager(AbsAgentManager):
        def train(self, experiences_by_agent, performance=None):
            self._assert_train_mode()

            # store experiences for each agent
            for agent_id, exp in experiences_by_agent.items():
                exp.update({"loss": [1e8] * len(list(exp.values())[0])})
                self.agent_dict[agent_id].store_experiences(exp)

            for agent in self.agent_dict.values():
                agent.train()

Main Loop with Actor and Learner (Single Process)
-------------------------------------------------

This single-process workflow of a learning policy's interaction with a MARO environment is comprised of:
- Initializing an environment with specific scenario and topology parameters.
- Defining scenario-specific components, e.g. shapers.
- Creating agents and an agent manager.
- Creating an `actor <https://maro.readthedocs.io/en/latest/key_components/rl_toolkit.html#learner-and-actor>`_ and a
`learner <https://maro.readthedocs.io/en/latest/key_components/rl_toolkit.html#learner-and-actor>`_ to start the
training process in which the agent manager interacts with the environment for collecting experiences and updating
policies.

.. code-block::python
    env = Env("cim", "toy.4p_ssdd_l0.0", durations=1120)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]
    state_shaper = CIMStateShaper(look_back=7, max_ports_downstream=2)
    action_shaper = CIMActionShaper(action_space=list(np.linspace(-1.0, 1.0, NUM_ACTIONS)))
    experience_shaper = TruncatedExperienceShaper(
        time_window=100, fulfillment_factor=1.0, shortage_factor=1.0, time_decay_factor=0.97
    )
    agent_manager = DQNAgentManager(
        create_dqn_agents(agent_id_list),
        state_shaper=state_shaper,
        action_shaper=action_shaper,
        experience_shaper=experience_shaper
    )

    scheduler = TwoPhaseLinearParameterScheduler(
        max_episode=100,
        parameter_names=["epsilon"],
        split_ep=50,
        start_values=0.4,
        mid_values=0.32,
        end_values=.0
    )

    actor = SimpleActor(env, agent_manager)
    learner = SimpleLearner(agent_manager, actor, scheduler)
    learner.learn()


Main Loop with Actor and Learner (Distributed/Multi-process)
--------------------------------------------------------------

We demonstrate a single-learner and multi-actor topology where the learner drives the program by telling remote actors
to perform roll-out tasks and using the results they sent back to improve the policies. The workflow usually involves
launching a learner process and an actor process separately. Because training occurs on the learner side and inference
occurs on the actor side, we need to create appropriate agent managers on both sides.

On the actor side, the agent manager must be equipped with all shapers as well as an explorer. Thus, The code for
creating an environment and an agent manager on the actor side is similar to that for the single-host version. As in the
single-process version, the environment and the agent manager are wrapped in a SimpleActor instance. To make the actor a
distributed worker, we need to further wrap it in an ActorWorker instance. Finally, we launch the worker and it starts to
listen to roll-out requests from the learner. The following code snippet shows the creation of an actor worker with a
simple(local) actor wrapped inside.

.. code-block:: python
    env = Env("cim", "toy.4p_ssdd_l0.0", durations=1120)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]
    agent_manager = DQNAgentManager(
        create_dqn_agents(agent_id_list),
        state_shaper=state_shaper,
        action_shaper=action_shaper,
        experience_shaper=experience_shaper
    )
    proxy_params = {
        "group_name": "distributed_cim", 
        "expected_peers": {"learner": 1}, 
        "redis_address": ("localhost", 6379),
        "max_retries": 15
    }
    actor_worker = ActorWorker(
        local_actor=SimpleActor(env=env, agent_manager=agent_manager),
        proxy_params=proxy_params
    )
    actor_worker.launch()

On the learner side, instead of creating an actor, we create an actor proxy and wrap it inside the learner. This proxy
serves as the communication interface for the learner and is responsible for sending roll-out requests to remote actor
processes and receiving results. Calling the train method executes the usual training loop except that the actual
roll-out is performed remotely. The code snippet below shows the creation of a learner with an actor proxy wrapped
inside that communicates with 3 actors. 

.. code-block:: python

    agent_manager = DQNAgentManager(
        create_dqn_agents(agent_id_list),
        state_shaper=state_shaper,
        action_shaper=action_shaper,
        experience_shaper=experience_shaper
    )
    proxy_params = {
        "group_name": "distributed_cim", 
        "expected_peers": {"actor": 3}, 
        "redis_address": ("localhost", 6379),
        "max_retries": 15
    }
    actor=ActorProxy(proxy_params=proxy_params, experience_collecting_func=concat_experiences_by_agent),
    scheduler = TwoPhaseLinearParameterScheduler(
        max_episode=100,
        parameter_names=["epsilon"],
        split_ep=50,
        start_values=0.4,
        mid_values=0.32,
        end_values=.0
    )
    learner = SimpleLearner(agent_manager, actor, scheduler)
    learner.learn()

.. note::

  All related code snippets are supported in `maro playground <https://hub.docker.com/r/arthursjiang/maro>`_.
