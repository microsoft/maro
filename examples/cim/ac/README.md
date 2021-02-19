# Overview

The CIM problem is one of the quintessential use cases of MARO. The example can
be run with a set of scenario configurations that can be found under
maro/simulator/scenarios/cim. General experimental parameters (e.g., type of
topology, type of algorithm to use, number of training episodes) can be configured
through config.yml. Each RL formulation has a dedicated folder, e.g., dqn, and
all algorithm-specific parameters can be configured through
the config.py file in that folder.

## Single-host Single-process Mode

To run the CIM example using the DQN algorithm under single-host mode, go to
examples/cim/dqn and run single_process_launcher.py. You may play around with
the configuration if you want to try out different settings.
