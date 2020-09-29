# Contributing to MARO

MARO is newborn for Reinforcement learning as a Service (RaaS) in the resource optimization domain. Your contribution is precious to make RaaS come true.

- [Open issues](https://github.com/microsoft/maro/issues) for reporting bugs and requesting new features.
- Contribute to [examples](https://github.com/microsoft/maro/tree/master/examples) to share your problem modeling to others.
- Contribute to [scenarios](https://github.com/microsoft/maro/tree/master/maro/simulator/scenarios) to provide more meaningful environments.
- Contribute to [topologies](https://github.com/microsoft/maro/tree/master/maro/simulator/scenarios/citi_bike/topologies) to enhance existing MARO scenarios.
- Contribute to [algorithms](https://github.com/microsoft/maro/tree/master/maro/rl/algorithms) to enrich MARO RL libraries.
- Contribute to [orchestration](https://github.com/microsoft/maro/tree/master/maro/cli) to broad MARO supported cloud services.
- Contribute to [communication](https://github.com/microsoft/maro/tree/master/maro/communication) to enhance MARO distributed training capacity.
- Contribute to [tests](https://github.com/microsoft/maro/tree/master/tests) to make it more reliable and stable.
- Contribute to [documentation](https://github.com/microsoft/maro/tree/master/maro) to make it straightforward for everyone.


# Note

Please make sure lint your code, and pass the code checking before pull request. 

We have prepared a configuration file for flake8 to lint.

```sh

# install flake8
pip install flake8

# lint with flake8
flake8 --config .github/linters/tox.ini

```
