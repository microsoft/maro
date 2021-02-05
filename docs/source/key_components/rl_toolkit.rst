
RL Toolkit
==========

MARO provides a full-stack abstraction for reinforcement learning (RL), which
empowers users to easily apply predefined and customized components to different
scenarios in a scalable way. The main abstractions include
`Learner, Actor <#learner-and-actor>`_\ , `Agent Manager <#agent-manager>`_\ ,
`Agent <#agent>`_\ , `Algorithm <#algorithm>`_\ ,
`State Shaper, Action Shaper, Experience Shaper <#shapers>`_\ , etc.

Learner and Actor
-----------------

.. image:: ../images/rl/overview.svg
   :target: ../images/rl/overview.svg
   :alt: RL Overview

* **Learner** is the abstraction of the learnable policy. It is responsible for
  learning a qualified policy to improve the business optimized object.

  .. code-block:: python

    # Train function of learner.
    def learn(self):
        for exploration_params in self._scheduler:
            performance, exp_by_agent = self._actor.roll_out(
                self._agent_manager.dump_models(),
                exploration_params=exploration_params
            )
            self._scheduler.record_performance(performance)
            self._agent_manager.train(exp_by_agent)

* **Actor** is the abstraction of experience collection. It is responsible for
  interacting with the environment and collecting experiences. The experiences
  collected during interaction will be used for the training of the learners.

  .. code-block:: python

    # Rollout function of actor.
    def roll_out(self, models=None, epsilons=None, seed: int = None):
        self._env.reset()

        # load models
        if model_dict is not None:
            self._agents.load_model(model_dict)

        # load exploration parameters:
        if exploration_params is not None:
            self._agents.set_exploration_params(exploration_params)

        metrics, decision_event, is_done = self._env.step(None)
        while not is_done:
            action = self._agents.choose_action(decision_event, self._env.snapshot_list)
            metrics, decision_event, is_done = self._env.step(action)
            self._agents.on_env_feedback(metrics)

        details = self._agents.post_process(self._env.snapshot_list) if return_details else None

        return self._env.metrics, details


Scheduler
---------

A ``Scheduler`` is the driver of an episodic learning process. The learner uses the scheduler to repeat the
rollout-training cycle a set number of episodes. For algorithms that require explicit exploration (e.g.,
DQN and DDPG), there are two types of schedules that a learner may follow:

* Static schedule, where the exploration parameters are generated using a pre-defined function of episode
  number. See ``LinearParameterScheduler`` and ``TwoPhaseLinearParameterScheduler`` provided in the toolkit
  for example.
* Dynamic schedule, where the exploration parameters for the next episode are determined based on the performance
  history. Such a mechanism is possible in our abstraction because the scheduler provides a ``record_performance``
  interface that allows it to keep track of roll-out performances.

Optionally, an early stopping checker may be registered if one wishes to terminate training when certain performance
requirements are satisfied, possibly before reaching the prescribed number of episodes.

Agent Manager
-------------

The agent manager provides a unified interactive interface with the environment
for RL agent(s). From the actor's perspective, it isolates the complex dependencies
of the various homogeneous/heterogeneous agents, so that the whole agent manager
will behave just like a single agent. Furthermore, to well serve the distributed algorithm
(scalable), the agent manager provides two kinds of working modes, which can be applied in
different distributed components, such as inference mode in actor, training mode in learner.

.. image:: ../images/rl/agent_manager.svg
   :target: ../images/rl/agent_manager.svg
   :alt: Agent Manager
   :width: 750

* In **inference mode**\ , the agent manager is responsible to access and shape
  the environment state for the related agent, convert the model action to an
  executable environment action, and finally generate experiences from the
  interaction trajectory.
* In **training mode**\ , the agent manager will optimize the underlying model of
  the related agent(s), based on the collected experiences from in the inference mode.

Agent
-----

The Agent is the kernel abstraction of the RL formulation for a real-world problem. 
Our abstraction decouples agent and its underlying model so that an agent can exist 
as an RL paradigm independent of the inner workings of the models it uses to generate 
actions or estimate values. For example, the actor-critic algorithm does not need to 
concern itself with the structures and optimizing schemes of the actor and critic models. 
This decoupling is achieved by the model abstraction described below. Ideally, the
agent is scenario agnostic because agent manager handles all scenario-specific logic.
In reality, however, this type of separation might not be achievable. 
 

.. image:: ../images/rl/agent.svg
   :target: ../images/rl/agent.svg
   :alt: Agent

.. code-block:: python

  class AbsAgent(ABC):
      def __init__(self, name: str, model: AbsLearningModel, config, experience_pool=None):
          self._name = name
          self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          self._model = model.to(self._device)
          self._config = config
          self._experience_pool = experience_pool


Block and LearningModel
--------------------------------

MARO provides an abstraction for the underlying models used by agents to form policies and estimate values.
The abstraction consists of ``AbsBlock`` and ``AbsLearningModel``, both of which subclass torch's nn.Module. 
The ``AbsBlock`` represents the smallest structural unit of an NN-based model. For instance, the ``FullyConnectedBlock`` 
provided in the toolkit is a stack of fully connected layers with features like batch normalization,
drop-out and skip connection. The ``AbsLearningModel`` is a collection of network components with
embedded optimizers and serves as an agent's "brain" by providing a unified interface to it. regardless of how many individual models it requires and how
complex the model architecture might be.

As an example, the initialization of the actor-critic algorithm may look like this:

.. code-block:: python

  actor_stack = FullyConnectedBlock(...)
  critic_stack = FullyConnectedBlock(...)
  model = SimpleMultiHeadModel(
      {"actor": actor_stack, "critic": critic_stack},
      optim_option={
        "actor": OptimizerOption(cls=Adam, params={"lr": 0.001})
        "critic": OptimizerOption(cls=RMSprop, params={"lr": 0.0001})  
      }
  )
  agent = ActorCritic("actor_critic", learning_model, config)

Choosing an action is simply:

.. code-block:: python

  model(state, task_name="actor", training=False)

And performing one gradient step is simply:

.. code-block:: python

  model.learn(critic_loss + actor_loss)


Explorer
-------

MARO provides an abstraction for exploration in RL. Some RL algorithms such as DQN and DDPG require
explicit exploration, the extent of which is usually determined by a set of parameters whose values
are generated by the scheduler. The ``AbsExplorer`` class is designed to cater to these needs. Simple
exploration schemes, such as ``EpsilonGreedyExplorer`` for discrete action space and ``UniformNoiseExplorer``
and ``GaussianNoiseExplorer`` for continuous action space, are provided in the toolkit.

As an example, the exploration for DQN may be carried out with the aid of an ``EpsilonGreedyExplorer``:

.. code-block:: python

  explorer = EpsilonGreedyExplorer(num_actions=10)
  greedy_action = learning_model(state, training=False).argmax(dim=1).data
  exploration_action = explorer(greedy_action)