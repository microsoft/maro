# Supply Chain Scenario

This README contains instructions for running the supply chain scenario using the scripts provided under ```examples/supply_chain/scripts```. For details on state, action and reward shaping based on MARO's supply chain business engine, refer to ```examples/supply_chain/sc_state_in_maro.md```.

The instructions require that you have Docker and Docker Compose installed and set up on your machine. For installing Docker, refer to https://docs.docker.com/get-docker/. For installing Docker Compose, refer to https://docs.docker.com/compose/install/. To run the supply chain scenario, go to ```examples/supply_chain/scripts``` and follow the steps below:
1. Run ```bash build.sh``` to build the docker images required for running the supply chain scenario. This only needs to be done once, unless changes are made to any of the source code in maro/maro except that in maro/maro/rl, which is mounted to the containers.
2. Execute ```bash run.sh``` to run the scenario in multiple containers. A docker-compose manifest yaml will be generated based on the value of ```num_actors``` in the "distributed" section of ```examples/supply_chain/config.yml```. The number of containers launched will be equal to this value plus 2 (one for the learner and one for the Redis server).
3. After the program terminates, execute ```bash kill.sh``` to clean up the containers created in Step 2.