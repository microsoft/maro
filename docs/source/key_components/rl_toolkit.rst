RL Toolkit
==========

MARO provides a full-stack abstraction for reinforcement learning (RL) which includes various customizable
components. In order to provide a gentle introduction for the RL toolkit, we cover the components in a top-down
manner, starting from the learning workflow.

Workflow
--------

The nice thing about MARO's RL workflows is that it is abstracted neatly from business logic, policies and learning algorithms,
making it applicable to practically any scenario that utilizes standard reinforcement learning paradigms. The workflow is
controlled by a main process that executes 2-phase learning cycles: roll-out and training (:numref:`1`). The roll-out phase
collects data from one or more environment simulators for training. There can be a single environment simulator located in the same thread as the main
loop, or multiple environment simulators running in parallel on a set of remote workers (:numref:`2`) if you need to collect large amounts of data
fast. The training phase uses the data collected during the roll-out phase to train models involved in RL policies and algorithms.
In the case of multiple large models, this phase can be made faster by having the computationally intensive gradient-related tasks
sent to a set of remote workers for parallel processing (:numref:`3`).

.. _1:
.. figure:: ../images/rl/learning_workflow.svg
   :alt: Overview
   :align: center

   Learning Workflow


.. _2:
.. figure:: ../images/rl/parallel_rollout.svg
   :alt: Overview
   :align: center

   Parallel Roll-out


.. _3:
.. figure:: ../images/rl/distributed_training.svg
   :alt: Overview
   :align: center

   Distributed Training


Environment Sampler
-------------------

An environment sampler is an entity that contains an environment simulator and a set of policies used by agents to
interact with the environment (:numref:`4`). When creating an RL formulation for a scenario, it is necessary to define an environment
sampler class that includes these key elements:

- how observations / snapshots of the environment are encoded into state vectors as input to the policy models. This
  is sometimes referred to as state shaping in applied reinforcement learning;
- how model outputs are converted to action objects defined by the environment simulator;
- how rewards / penalties are evaluated. This is sometimes referred to as reward shaping.

In parallel roll-out, each roll-out worker should have its own environment sampler instance.


.. _4:
.. figure:: ../images/rl/env_sampler.svg
   :alt: Overview
   :align: center

   Environment Sampler


Policy
------

``Policy`` is the most important concept in reinforcement learning. In MARO, the highest level abstraction of a policy
object is ``AbsPolicy``. It defines the interface ``get_actions()`` which takes a batch of states as input and returns
corresponding actions.
The action is defined by the policy itself. It could be a scalar or a vector or any other types.
Env sampler should take responsibility for parsing the action to the acceptable format before passing it to the
environment.

The simplest type of policy is ``RuleBasedPolicy`` which generates actions by pre-defined rules. ``RuleBasedPolicy``
is mostly used in naive scenarios. However, in most cases where we need to train the policy by interacting with the
environment, we need to use ``RLPolicy``. In MARO's design, a policy cannot train itself. Instead,
polices could only be trained by :ref:`trainer` (we will introduce trainer later in this page). Therefore, in addition
to ``get_actions()``, ``RLPolicy`` also has a set of training-related interfaces, such as ``step()``, ``get_gradients()``
and ``set_gradients()``. These interfaces will be called by trainers for training. As you may have noticed, currently
we assume policies are built upon deep learning models, so the training-related interfaces are specifically
designed for gradient descent.


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

The above code snippet creates a ``ValueBasedPolicy`` object. Let's pay attention to the parameter ``q_net``.
``q_net`` accepts a ``DiscreteQNet`` object, and it serves as the core part of a ``ValueBasedPolicy`` object. In
other words, ``q_net`` defines the model structure of the Q-network in the value-based policy, and further determines
the policy's behavior. ``DiscreteQNet`` is an abstract class, and ``MyQNet`` is a user-defined implementation
of ``DiscreteQNet``. It can be a simple MLP, a multi-head transformer, or any other structure that the user wants.

MARO provides a set of abstractions of basic & commonly used PyTorch models like ``DiscereteQNet``, which enables
users to implement their own deep learning models in a handy way. They are:

- ``DiscreteQNet``: For ``ValueBasedPolicy``.
- ``DiscretePolicyNet``: For ``DiscretePolicyGradient``.
- ``ContinuousPolicyNet``: For ``ContinuousPolicyGradient``.

Users should choose the proper types of models according to the type of policies, and then implement their own
models by inheriting the abstract ones (just like ``MyQNet``).

There are also some other models for training purposes. For example:

- ``VNet``: Used in the critic part in the actor-critic algorithm.
- ``MultiQNet``: Used in the critic part in the MADDPG algorithm.
- ...

The way to use these models is exactly the same as the way to use the policy models.

.. _trainer:

Algorithm (Trainer)
-------

When introducing policies, we mentioned that policies cannot train themselves. Instead, they have to be trained
by external algorithms, which are also called trainers.
In MARO, a trainer represents an RL algorithm, such as DQN, actor-critic,
and so on. These two concepts are equivalent in the MARO context.
Trainers take interaction experiences and store them in the internal memory, and then use the experiences
in the memory to train the policies. Like ``RLPolicy``, trainers are also concrete classes, which means they could
be used by configuring parameters. Currently, we have 4 trainers (algorithms) in MARO:

- ``DiscreteActorCriticTrainer``: Actor-critic algorithm for policies that generate discrete actions.
- ``DiscretePPOTrainer``: PPO algorithm for policies that generate discrete actions.
- ``DDPGTrainer``: DDPG algorithm for policies that generate continuous actions.
- ``DQNTrainer``: DQN algorithm for policies that generate discrete actions.
- ``DiscreteMADDPGTrainer``: MADDPG algorithm for policies that generate discrete actions.

Each trainer has a corresponding ``Param`` class to manage all related parameters. For example,
``DiscreteActorCriticParams`` contains all parameters used in ``DiscreteActorCriticTrainer``:

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

   DiscreteActorCriticTrainer(
       name='ac',
       params=DiscreteActorCriticParams(
           get_v_critic_net_func=lambda: MyCriticNet(state_dim=128),
           reward_discount=.0,
           grad_iters=10,
           critic_loss_cls=torch.nn.SmoothL1Loss,
           min_logp=None,
           lam=.0
       )
   )

In order to indicate which trainer each policy is trained by, in MARO, we require that the name of the policy
start with the name of the trainer responsible for training it. For example, policy ``ac_1.policy_1`` is trained
by the trainer named ``ac_1``. Violating this provision will make MARO unable to correctly establish the
corresponding relationship between policy and trainer.

More details and examples can be found in the code base (`link`_).

.. _link: https://github.com/microsoft/maro/blob/master/examples/rl/cim/policy_trainer.py

As a summary, the relationship among policy, model, and trainer is demonstrated in :numref:`5`:

.. _5:
.. figure:: ../images/rl/policy_model_trainer.svg
   :alt: Overview
   :align: center

   Summary of policy, model, and trainer
