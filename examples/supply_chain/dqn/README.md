Running the supply chain scenario consists of 3 simple steps:

1. To build the docker images required for running the supply chain scenario, go to examples/suuply_chain/scripts and run ```bash build.sh```. This step only needs to be done once, unless changes are made to the source code in maro/maro.
2. Execute ```bash run.sh``` to run the scenario in multiple containers. A docker-compose manifest yaml will be generated based on the value of ```num_actors``` in the ```config.yml```. The number of containers launched will be equal to this value plus 2 (one for the learner and one for the Redis server).
3. After the program terminates, execute ```bash kill.sh``` to clean up the containers launched in Step 2.