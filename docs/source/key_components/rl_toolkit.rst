
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
    def learn(self, total_episodes):
        for exploration_params in self._scheduler:
            performance, exp_by_agent = self._actor.roll_out(
                model_dict=None if self._is_shared_agent_instance() else self._agent_manager.dump_models(),
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
            self._agents.load_models(model_dict)

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

A ``Scheduler`` is an iterable object responsible for driving an episodic learning process and recording 
roll-out performances. In the simplest case, it repeates the rollout-training cycle a set number of episodes. 
Optionally, a callable early stopping checker can be registered if one wishes to terminate learning as soon
as some conditions are met. For algorithms that require explicit exploration (e.g., DQN and DDPG), users can 
implement the ``get_next_exploratin_params`` interface, so that the scheduler will generate exploration 
parameter values for each episode. The generated values can either follow a pre-defined schedule, such as
the ``LinearParameterScheduler`` and ``TwoPhaseLinearParameterScheduler`` provided in the toolkit, or a dynamic
mechanism where the next values are determined based on the performance history.      


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

An agent is a combination of (RL) algorithm, experience pool, and a set of
non-algorithm-specific parameters (algorithm-specific parameters are managed by
the algorithm module). Non-algorithm-specific parameters are used to manage
experience storage, sampling strategies, and training strategies. Since all kinds
of scenario-specific stuff will be handled by the agent manager, the agent is
scenario agnostic.

.. image:: ../images/rl/agent.svg
   :target: ../images/rl/agent.svg
   :alt: Agent

.. code-block:: python

  class AbsAgent(ABC):
      def __init__(self, name: str, algorithm: AbsAlgorithm, experience_pool: AbsStore = None):
        self._name = name
        self._algorithm = algorithm
        self._experience_pool = experience_pool


Algorithm
---------

The algorithm is the kernel abstraction of the RL formulation for a real-world problem. The 
``LearningModule`` and ``LearningModuleManager`` abstractions described below allow an algorithm
to be abstracted as the simple combination of a model (LearningModuleManager) and a configuration 
object.  


.. image:: ../images/rl/algorithm.svg
   :target: ../images/rl/algorithm.svg
   :alt: Algorithm
   :width: 650

* ``choose_action`` is used to make a decision based on a provided model state.
* ``train`` is used to trigger training and the policy update from external.

.. code-block:: python

  class AbsAlgorithm(ABC):
      def __init__(self, model: LearningModuleManager, config):
          self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          self._model = model.to(self._device)
          self._config = config


Block, LearningModule and LearningModuleManager
-----------------------------------------------

MARO provides an abstraction for the underlying models used by agents to form policies and estimate values.
The abstraction consists of 3-level hierachy formed from the bottom up by ``AbsBlock``, ``LearningModule`` 
and ``LearningModuleManager``, all of which subclass torch's nn.Module. Conceptually, an ``AbsBlock`` is the 
smallest structural unit of an NN-based model. For instance, the ``FullyConnectedBlock`` provided in the toolkit 
represents a stack of fully connected layers with features like batch normalization, drop-out and skip connection. 
A ``LearningModule`` consists of one or more such blocks, as well as an optimizer responsible for applying gradient 
steps to the trainable parameters of these blocks. Therefore, a ``LearningModule`` represents the smallest trainable 
unit of a model. Finally, the complete model as used directly by an ``Algorithm`` is represented as a ``LearningModuleManager``, 
an abstraction that entails multi-task learning that is common in RL but presents a unified interface to the 
algorithm. A ``LearningModuleManager`` consists of one or more task modules as "heads" and an optional shared 
module at the bottom, which serves to produce a representation as input to all task modules. 

.. image:: ../images/rl/learning_model.svg
   :target: ../images/rl/learning_model.svg
   :alt: Algorithm
   :width: 650

For intance, the initialization of the actor-critic algorithm may look like this:

.. code-block:: python

  actor_module = LearningModule(name="actor", block_list=..., optiimizer_options=...)
  critic_module = LearningModule(name="critic", block_list=..., optiimizer_options=...)
  
  actor_critic = ActorCritic(LearningModuleManager(actor_module, critic_module), config)

Choosing an action is simply:

.. code-block:: python

  self._model(state, task_name="actor", is_training=False)

And performing one gradient step is simply:

.. code-block:: python

  self._model.learn(critic_loss + actor_loss)


Explorer
-------

MARO provides an abstraction for exploration in RL. Some RL algorithms such as DQN and DDPG require 
external perturbations to model-generated actions to explore trajectory search space. The extent of 
these perturbations usually determined by a set of parameters whose values are generated by the scheduler.
The ``AbsExplorer`` class defines ``set_parameters`` and ``__call__`` methods to cater to these needs. 
The ``set_parameters`` method sets the exploration parameters to the values generated by the scheduler, 
while the ``__call__`` method perturbs a model-generated action to obtain an exploratory action. Simple
exploration schemes, such as ``EpsilonGreedyExplorer`` for discrete action space and ``UniformNoiseExplorer`` 
and ``GaussianNoiseExplorer`` for continuous action space, are provided in the toolkit. Users are free to 
implement their own exploration logic by subclassing ``AbsExplorer`` and implementing the ``set_parameters`` 
and ``__call__`` methods. 
