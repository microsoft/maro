RL Toolkit
==========

MARO provides a full-stack abstraction for reinforcement learning (RL) which includes various customizable
components. In order to provide a gentle introduction for the RL toolkit, we cover the components in a top-down
manner, starting from the learning workflow.

Workflow
--------

The nice thing about MARO's RL workflows is that it is abstracted neatly from business logic, policies and learning algorithms,
making it applicable to practically any scenario that utilizes standard reinforcement learning paradigms. The workflow is
controlled by a main process that executes 2-phase learning cycles that consist of roll-out and training.
The roll-out phase collects data from one or more environment simulators for training. There can be a single environment
simulator located in the same thread as the main loop, or multiple environment simulators distributed amongst a set of
remote workers for parallelism if you need to collect a large amount of data fast. The training phase uses the data
collected during the roll-out phase to train models involved in RL policies and algorithms. It can also be local, where trainers update their models
in a single thread in a sequential mannner, or distributed, where 


The transition from one phase to the other is synchrounous. To handle slow roll-out workers, the
roll-out manager can be configured to pass the results from a subset of roll-out workers (i.e., the faster ones) to the
policy manager. On the other hand, the policy manager always waits until all policies are updated before passing the
policy states to the roll-out manager.


.. figure:: ../images/rl/learning_cycle.svg
   :alt: Overview
   
   Synchronous Learning Cycle


.. figure:: ../images/rl/rollout_manager.svg
   :alt: Overview

   Roll-out Manager


Environment Sampler
-------------------

It is necessary to implement an environment sampler (a subclass of ``AbsEnvSampler``) with user-defined state, action
and reward shaping to collect roll-out information for learning and testing purposes. An environment sampler can be
easily turned into a roll-out worker or an actor for synchronous and asynchronous learning, respectively.


.. figure:: ../images/rl/env_sampler.svg
   :alt: Overview

   Environment Sampler


Policy
------

``Policy`` is the most important concept in reinforcement learning. In MARO, the highest level abstraction of a policy
object is ``AbsPolicy``. It defines the interface ``get_actions()`` which takes a batch of states a inputs and returns
corresponding actions.

The simplest type of policy is ``RuleBasedPolicy`` which generates actions by pre-defined rules. ``RuleBasedPolicy``
is mostly used in naive scenarios. However, in most cases, we need to train the policy by interacting with the
environment, these are the cases we need to use ``RLPolicy``. In MARO's design, a policy cannot train itself. Instead,
polices could only be trained by :ref:`trainer` (we will introduce trainer later in this page). Therefore, in addition
to ``get_actions``, ``RLPolicy`` also has a set of training-related interfaces, such as ``step()``, ``get_gradients()``
and ``set_gradients()``. These interfaces will be called by trainers for training. As you may noticed, currently
we assume policies are built upon deep learning models, so the training-related interfaces are specifically
designed for gradient decent.

``RLPolicy`` is further divided into three types:
- ``ValueBasedPolicy``: For valued-based policies.
- ``DiscretePolicyGradient``: For gradient-based policies that generate discrete actions.
- ``ContinuousPolicyGradient``: For gradient-based policies that generate continuous actions.

The above classes are all concrete classes. Users do not need to implement any new classes, but can directly
create a policy object by configuring parameters. Here is a simple example:

.. code-block:: python

   ValueBasedPolicy(
       name="policy",
       q_net=MyQNet(state_dim=128, action_num=64),
   )


For now, you may have no idea about the ``q_net`` parameter, but don't worry, we will introduce it in the next section.

Model
-----

The above code snippet creates a ``ValueBasedPolicy`` object. Let's pay our attention to the parameter ``q_net``.
``q_net`` accepts a ``DiscreteQNet`` object, and it serves as the core part of a ``ValueBasedPolicy`` object. In
other words, ``q_net`` defines the model structure of the Q-network in the value-based policy, and further determines
the policy's behavior. ``DiscreteQNet`` is an abstract class, and ``MyQNet`` is one of the user-defined implementation
of ``DiscreteQNet``. It can be a simple MLP, a multihead transformer, or any other structure that the user wants.

MARO provides a set of abstractions of basic & commonly used PyTorch models like ``DiscereteQNet``, which enables
users to implement their own deep learning models in a handy way. They are:

- ``DiscreteQNet``: For ``ValueBasedPolicy``.
- ``DiscretePolicyNet``: For ``DiscretePolicyGradient``.
- ``ContinuousPolicyNet``: For ``ContinuousPolicyGradient``.

Users should choose the proper types of models according to the type of policies, and then implement their own
models by inherit the abstract ones (just like ``MyQNet``).

There are also some other models for training purpose. For example:

- ``VNet``: Used in the critic part in the actor-critic algorithm.
- ``MultiQNet``: Used in the critic part in the MADDPG algorithm.
- ...

The way to use these models is exactly the same as the way to use the policy models.

.. _trainer:

Trainer
-------

When introducing policies, we mentioned that policies cannot train themselves. Instead, they have to be trained
by external trainers. In MARO, a trainer is corresponding to a kind of RL algorithm, such as DQN, actor-critic,
and so on. Trainers take interaction experiences and store them in a internal memory, and then use the experiences
in the memory to train the policies. Like ``RLPolicy``, trainers are also concrete classed, which means they could
be used by configuring parameters. Currently, we have 4 trainers in MARO:

- ``DiscreteActorCritic``: Actor-critic algorithm for policies that generate discrete actions.
- ``DDPG``: DDPG algorithm for policies that generate continuous actions.
- ``DQN``: DQN algorithm for policies that generate discrete actions.
- ``DiscreteMADDPG``: MADDPG algorithm for policies that generate discrete actions.

Each trainer has a corresponding ``Param`` class that used to manage all related parameters. For example,
``DiscreteActorCriticParams`` contains all parameters used in ``DiscreteActorCritic``:

.. code-block:: python

   @dataclass
   class DiscreteActorCriticParams(TrainerParams):
       get_v_critic_net_func: Callable[[], VNet] = None
       reward_discount: float = 0.9
       grad_iters: int = 1
       critic_loss_cls: Callable = None
       clip_ratio: float = None
       lam: float = 0.9
       min_logp: Optional[float] = None

An example of creating an actor-critic trainer:

.. code-block:: python

   DiscreteActorCritic(
       name='ac',
       params=DiscreteActorCriticParams(
           device="cpu",
           get_v_critic_net_func=lambda: MyCriticNet(state_dim=128),
           reward_discount=.0,
           grad_iters=10,
           critic_loss_cls=torch.nn.SmoothL1Loss,
           min_logp=None,
           lam=.0
       )
   )

In order to indicate which trainer each policy is trained by, in MARO, we require that the name of the policy
starts with the name of the trainer responsible for training it. For example, policy ``ac_1.policy_1`` is trained
by the trainer called ``ac_1``. Violating this provision will make MARO unable to correctly establish the
corresponding relationship between policy and trainer.

More details and examples can be found in the code base.
