
RL Toolkit
==========

MARO provides a full-stack abstraction for reinforcement learning (RL), which enables users to
apply predefined and customized components to various scenarios. The main abstractions include
fundamental components such as `Agent <#agent>`_\ and `Shaper <#shaper>`_\ , and training routine
controllers such as `Actor <#actor>`_\ and `Learner <#learner>`_.


Policy
------

A policy is used by an agent to decide what action to take given an observation of the environment.
Accordingly, the abstract ``AbsPolicy`` class exposes a ``choose_action`` interface. This abstraction
can encompass both static policies, such as rule-based policies, and updatable policies, such as RL
policies. The latter is abstracted through the ``AbsCorePolicy`` sub-class which also exposes a ``update``
interface. By default, updatable policies require an experience manager to store and retrieve simulation
data (in the form of "experiences") based on which updates can be made.


.. image:: ../images/rl/agent.svg
   :target: ../images/rl/agent.svg
   :alt: Agent

.. code-block:: python

  class AbsPolicy(ABC):
      @abstractmethod
      def choose_action(self, state):
          raise NotImplementedError


  class AbsCorePolicy(AbsPolicy):
      def __init__(self, experience_manager: AbsExperienceManager):
          super().__init__()
          self.experience_manager = experience_manager

      @abstractmethod
      def update(self):
          raise NotImplementedError


Core Model
----------

In the deep reinforcement learning (DRL) world, a core policy usually includes one or more neural-network-based models,
which may be used to compute action preferences or estimate state / action values. The core model abstraction is designed
to decouple the the inner workings of these models from the algorithmic aspects of the policy that uses them. For example,
the actor-critic algorithm does not need to concern itself with the structures and optimizing schemes of the actor and
critic models. The abstraction consists of ``AbsBlock`` and ``AbsCoreModel``, both of which subclass torch's nn.Module.
The ``AbsBlock`` represents the smallest structural unit of an NN-based model. For instance, the ``FullyConnectedBlock``
is a stack of fully connected layers with features like batch normalization, drop-out and skip connection.
The ``AbsCoreModel`` is a collection of network components with embedded optimizers. Several classes are designed  

As an example, the initialization of the actor-critic algorithm may look like this:

.. code-block:: python

  actor_stack = FullyConnectedBlock(...)
  critic_stack = FullyConnectedBlock(...)
  ac_model = SimpleMultiHeadModel(
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


Exploration
-----------

Some RL algorithms such as DQN and DDPG require explicit exploration governed by a set of parameters. The
``AbsExploration`` class is designed to cater to these needs. Simple exploration schemes, such as ``EpsilonGreedyExploration`` for discrete action space
and ``UniformNoiseExploration`` and ``GaussianNoiseExploration`` for continuous action space, are provided in
the toolkit.

As an example, the exploration for DQN may be carried out with the aid of an ``EpsilonGreedyExploration``:

.. code-block:: python

  exploration = EpsilonGreedyExploration(num_actions=10)
  greedy_action = learning_model(state, training=False).argmax(dim=1).data
  exploration_action = exploration(greedy_action)


Tools for Training
------------------

.. image:: ../images/rl/learner_actor.svg
   :target: ../images/rl/learner_actor.svg
   :alt: RL Overview

The RL toolkit provides tools that make local and distributed training easy:
* Learner, the central controller of the learning process, which consists of collecting simulation data from
  remote actors and training the agents with them. The training data collection can be done in local or
  distributed fashion by loading an ``Actor`` or ``ActorProxy`` instance, respectively.  
* Actor, which implements the ``roll_out`` method where the agent interacts with the environment for one
  episode. It consists of an environment instance and an agent (a single agent or multiple agents wrapped by
  ``MultiAgentWrapper``). The class provides the worker() method which turns it to an event loop where roll-outs
  are performed on the learner's demand. In distributed RL, there are typically many actor processes running
  simultaneously to parallelize training data collection.
* Actor proxy, which also implements the ``roll_out`` method with the same signature, but manages a set of remote
  actors for parallel data collection.
* Trajectory, which is primarily responsible for translating between scenario-specific information and model
  input / output. It implements the following methods which are used as callbacks in the actor's roll-out loop: 
  * ``get_state``, which converts observations of an environment into model input. For example, the observation
    may be represented by a multi-level data structure, which gets encoded by a state shaper to a one-dimensional
    vector as input to a neural network. The state shaper usually goes hand in hand with the underlying policy
    or value models. 
  * ``get_action``, which provides model output with necessary context so that it can be executed by the
    environment simulator.
  * ``get_reward``, which computes a reward for a given action.
  * ``on_env_feedback``, which defines things to do upon getting feedback from the environment.  
  * ``on_finish``, which defines things to do upon completion of a roll-out episode.
