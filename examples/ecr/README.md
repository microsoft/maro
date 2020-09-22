This page contains instructions on how to use MARO on the Empty Container Repositioning (ECR) problem

### overview
The ECR problem is one of the quintessential use cases of MARO. The example can be run with a set of artificial scenario 
configurations that can be found under maro/simulator/scenarios/ecr. General experimental parameters (e.g., type of 
topology, type of algorithm to use, number of training episodes) can be configured through config.yml. Each RL algorithm
has a dedicated folder, e.g., q_learning, and all algorithm-specific parameters can be configured through the config.py 
file in that folder.   

### Single host mode:
To run the ECR example under single-host mode, go to single_host_mode and execute launcher.py.

### Mock distributed mode:
To simulate distributed ECR using multiple processes, go to distributed_mode and execute mock_launcher.py
   