
RL Toolkit
==========

MARO provides a full-stack abstraction for reinforcement learning (RL), which enables users to
apply predefined and customized components to various scenarios. The main abstractions include
fundamental components such as `Agent <#agent>`_\ and `Shaper <#shaper>`_\ , and training routine
controllers such as `Actor <#actor>`_\ and `Learner <#learner>`_.


Policy
------

A policy is a an agent's mechanism to determine an action to take given an observation of the environment.
Accordingly, the abstract ``AbsPolicy`` class exposes a ``choose_action`` interface. This abstraction encompasses
both static policies, such as rule-based policies, and updatable policies, such as RL policies. The latter is
abstracted through the ``AbsCorePolicy`` sub-class which also exposes a ``update`` interface. By default, updatable
policies require an experience manager to store and retrieve simulation data (in the form of "experiences sets") based
on which updates can be made.


.. image:: ../images/rl/agent.svg
   :target: ../images/rl/agent.svg
   :alt: Agent

.. code-block:: python

  class AbsPolicy(ABC):
      @abstractmethod
      def choose_action(self, state):
          raise NotImplementedError


  class AbsCorePolicy(AbsPolicy):
      def __init__(self, experience_memory: ExperienceMemory):
          super().__init__()
          self.experience_memory = experience_memory

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
is a stack of fully connected layers with features like batch normalization, drop-out and skip connection. The ``AbsCoreModel``
is a collection of network components with embedded optimizers. Subclasses of ``AbsCoreModel`` provided for use with specific
RL algorithms include ``DiscreteQNet`` for DQN, ``DiscretePolicyNet`` for Policy Gradient, ``DiscreteACNet`` for Actor-Critic
and ``ContinuousACNet`` for DDPG.

The code snippet below shows how to create a model for the actor-critic algorithm with a shared bottom stack:

.. code-block:: python

  class MyACModel(DiscreteACNet):
      def forward(self, states, actor=True, critic=True):
          features = self.component["representation"](states)
          return (
              self.component["actor"](features) if actor else None,
              self.component["critic"](features) if critic else None
          )


  representation_stack = FullyConnectedBlock(...)
  actor_head = FullyConnectedBlock(...)
  critic_head = FullyConnectedBlock(...)
  ac_model = SimpleMultiHeadModel(
      {"representation": representation_stack, "actor": actor_head, "critic": critic_head},
      optim_option={
        "representation": OptimizerOption(cls="adam", params={"lr": 0.0001}),
        "actor": OptimizerOption(cls="adam", params={"lr": 0.001}),
        "critic": OptimizerOption(cls="rmsprop", params={"lr": 0.0001})  
      }
  )

To generate stochastic actions given a batch of states, call ``get_action`` on the model instance: 

.. code-block:: python

  action, log_p = ac_model.get_action(state)

To performing a single gradient step on the model, call the ``step`` function: 

.. code-block:: python

  ac_model.step(critic_loss + actor_loss)

Here it is assumed that the losses have been computed using the same model instance and the gradients have
been generated for the internal components.  


Experience
----------

An ``ExperienceSet`` is a synonym for training data for RL policies. The data originate from the simulator and
get processed and organized into a set of transitions in the form of (state, action, reward, next_state, info),
where ''info'' contains information about the transition that is not encoded in the state but may be necessary
for sampling purposes. An ``ExperienceMemory`` is a storage facility for experience sets and is maintained by
a policy for storing and retrieving training data. Sampling from the experience memory can be customized by 
registering a user-defined sampler to it.  


Exploration
-----------

Some RL algorithms such as DQN and DDPG require explicit exploration governed by a set of parameters. The
``AbsExploration`` class is designed to cater to these needs. Simple exploration schemes, such as ``EpsilonGreedyExploration`` for discrete action space
and ``UniformNoiseExploration`` and ``GaussianNoiseExploration`` for continuous action space, are provided in
the toolkit.

As an example, the exploration for DQN may be carried out with the aid of an ``EpsilonGreedyExploration``:

.. code-block:: python

  exploration = EpsilonGreedyExploration(num_actions=10)
  greedy_action = q_net.get_action(state)
  exploration_action = exploration(greedy_action)


Environment Wrapper
-------------------

An environment wrapper a wrapper for a raw MARO environment that provides unified interfaces to the external
RL workflow through user-defined state, action and reward shaping. It is also responsible for caching transitions and preparing experiences
for training. It is necessary to implement an environment wrapper for the environemtn of your choice in order to
run the RL workflow using the training tools described below. Key methods that need to be defined for an environment
wrapper include:
* ``get_state``, which converts observations of an environment into model input. For example, the observation
may be represented by a multi-level data structure, which gets encoded to a one-dimensional vector as input to
a neural network.
* ``to_env_action``, which provides model output with necessary context so that it can be executed by the
environment simulator.
* ``get_reward``, for evaluating rewards.


Tools for Training
------------------

.. image:: ../images/rl/learner_actor.svg
   :target: ../images/rl/learner_actor.svg
   :alt: RL Overview

The RL toolkit provides tools that make local and distributed training easy:
* Learner, which consists of a roll-out manager and a policy manager, is the controller for a learning process.
The learner process executes training cycles that alternate between data collection and policy updates.   
* Rollout manager, which is respnsible for collecting simulation data. The ``LocalRolloutManager`` performs roll-outs
locally, while the ``ParallelRolloutManager`` manages a set of remote ``Actor``s to collect simulation data in
parallel.
* Actor, which consists of an environment instance and a set of policies that agents use to interact with it, is a
remote roll-out worker instance managed by a ``ParallelRolloutManager``.
* Policy manager, which manages a set of policies and controls their updates. The policy instances may reside in the
manager (``LocalPolicyManager``) or be distributed on a set of remote nodes (``ParallelPolicyManager``, to be implemented).
