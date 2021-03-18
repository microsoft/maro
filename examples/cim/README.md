# Container Inventory Management

Container inventory management (CIM) is a scenario where reinforcement learning (RL) can potentially prove useful. Three algorithms are used to learn the multi-agent policy in given environments. Each algorithm has a ``config`` folder which contains ``agent_config.py`` and ``training_config.py``. The former contains parameters for the underlying models and algorithm specific hyper-parameters. The latter contains parameters for the environment and the main training loop. The file ``common.py`` contains parameters and utility functions shared by some or all of these algorithms. 

In the ``ac`` folder, , the policy is trained using the Actor-Critc algorithm in single-threaded fashion. The example can be run by simply executing ``python3 main.py``. Logs will be saved in a file named ``cim-ac.CURRENT_TIME_STAMP.log`` under the ``ac/logs`` folder, where ``CURRENT_TIME_STAMP`` is the time of executing the script. 

In the ``dqn`` folder, the policy is trained using the DQN algorithm in multi-process / distributed mode. This example can be run in three ways. 
* ``python3 main.py`` or ``python3 main.py -w 0`` runs the example in multi-process mode, in which a main process spawns one learner process and a number of actor processes as specified in ``config/training_config.py``.
* ``python3 main.py -w 1`` launches the learner process only. This is for distributed training and expects a number of actor processes (as specified in ``config/training_config.py``) running on some other node(s).
* ``python3 main.py -w 2`` launches the actor process only. This is for distributed training and expects a learner process running on some other node.
Logs will be saved in a file named ``GROUP_NAME.log`` under the ``{ac_gnn, dqn}/logs`` folder, where ``GROUP_NAME`` is specified in the "group" field in ``config/training_config.py``.
