# Container Inventory Management Using GNN-based Actor_Critic (Multi-process / Distributed)

This example demonstrates a combination of the graph neural network (GNN) and the actor-critic algorithm applied to the container inventory management (CIM) scenario in multi-process / distributed mode, where inference is performed on the learner side. The example can be run in three ways. 
* ``python3 main.py`` or ``python3 main.py -w 0`` runs the example in multi-process mode, in which a main process spawns one learner process and a number of actor processes as specified in ``config/training_config.py``.
* ``python3 main.py -w 1`` launches the learner process only. This is for distributed training and expects a number of actor processes (as specified in ``config/training_config.py``) running on some other node(s).
* ``python3 main.py -w 2`` launches the actor process only. This is for distributed training and expects a learner process running on some other node.

Logs generated during a run will be saved in a file named ``GROUP_NAME.log`` under the ``logs`` folder, where ``GROUP_NAME`` is specified in th "group" field in ``config/training_config.py``.