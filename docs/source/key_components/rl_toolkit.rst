
RL Toolkit
==========

MARO provides a full-stack abstraction for reinforcement learning (RL), which enables users to
apply predefined and customized components to various scenarios. The main abstractions include
fundamental components such as `Agent <#agent>`_\ and `Shaper <#shaper>`_\ , and training routine
controllers such as `Actor <#actor>` and `Learner <#learner>`.


Agent
-----

The Agent is the kernel abstraction of the RL formulation for a real-world problem. 
Our abstraction decouples agent and its underlying model so that an agent can exist 
as an RL paradigm independent of the inner workings of the models it uses to generate 
actions or estimate values. For example, the actor-critic algorithm does not need to 
concern itself with the structures and optimizing schemes of the actor and critic models. 
This decoupling is achieved by the Core Model abstraction described below.


.. image:: ../images/rl/agent.svg
   :target: ../images/rl/agent.svg
   :alt: Agent

.. code-block:: python

  class AbsAgent(ABC):
      def __init__(self, model: AbsCoreModel, config, experience_pool=None):
          self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          self.model = model.to(self.device)
          self.config = config
          self._experience_pool = experience_pool


Core Model
----------

MARO provides an abstraction for the underlying models used by agents to form policies and estimate values.
The abstraction consists of ``AbsBlock`` and ``AbsCoreModel``, both of which subclass torch's nn.Module. 
The ``AbsBlock`` represents the smallest structural unit of an NN-based model. For instance, the ``FullyConnectedBlock`` 
provided in the toolkit is a stack of fully connected layers with features like batch normalization,
drop-out and skip connection. The ``AbsCoreModel`` is a collection of network components with
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
--------

MARO provides an abstraction for exploration in RL. Some RL algorithms such as DQN and DDPG require
explicit exploration governed by a set of parameters. The ``AbsExplorer`` class is designed to cater
to these needs. Simple exploration schemes, such as ``EpsilonGreedyExplorer`` for discrete action space
and ``UniformNoiseExplorer`` and ``GaussianNoiseExplorer`` for continuous action space, are provided in
the toolkit.

As an example, the exploration for DQN may be carried out with the aid of an ``EpsilonGreedyExplorer``:

.. code-block:: python

  explorer = EpsilonGreedyExplorer(num_actions=10)
  greedy_action = learning_model(state, training=False).argmax(dim=1).data
  exploration_action = explorer(greedy_action)


Tools for Distributed Training
------------------------------

.. image:: ../images/rl/learner_actor.svg
   :target: ../images/rl/learner_actor.svg
   :alt: RL Overview

The RL toolkit provides tools that make distributed training easy:
* Learner, the central controller of the training process in a distributed setting. Its task is to
  collect training data from remote actors and train the agents with it. There are two ways of doing
  so: 1) sending each actor a copy of the current model so that they can make action decisions on their
  own; 2) providing action decisions directly to actors (https://arxiv.org/pdf/1910.06591.pdf).  
* Actor, which implements the ``roll_out`` method where the agent interacts with the environment
  for one episode. It consists of an environment instance and an agent (a single agent or multiple agents
  wrapped by ``MultiAgentWrapper``). It is usually necessary to define shaping functions that perform
  translations between scenario-specific information and model input / output. Three types of shaping
  are often necessary: 
  * State shaping, which converts observations of an environment into model input. For example, the observation
    may be represented by a multi-level data structure, which gets encoded by a state shaper to a one-dimensional
    vector as input to a neural network. The state shaper usually goes hand in hand with the underlying policy
    or value models. 
  * Action shaping, which provides model output with necessary context so that it can be executed by the
    environment simulator.
  * Reward shaping, which computes a reward for a given action.
  The class provides the as_worker() method which turns it to an event loop where roll-outs are performed
  on the learner's demand. In distributed RL, there are typically many actor processes running
  simultaneously to parallelize training data collection.
* Decision client, which communicates with the remote learner to obtain action decisions on behalf of
  the actor.
