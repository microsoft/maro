# Brief tutorial to try MARO Supply Chain

## Source Code and Environment

### Source Code

You can get the source code from the GitHub: https://github.com/microsoft/maro.

```sh
git clone https://github.com/microsoft/maro.git
```

Currently, we suggest you to try the code of branch **sc_tutorial**.

### Docker

You can pull the docker image to have a quick try by:

```sh
docker pull maro2020/maro_sc
```

Then run the given script to start the container:

```sh
bash scripts/run_sc_playground.sh
```

Now your local *examples/*, *notebooks/*, *topologies/* (of supply_chain scenario only) are bind mounted into the docker container you just started. Also, there is a jupyter lab running in port 40010, access *http://localhost:40010/* to have a try :)

### Set Up the Environment Locally

You can also set up the environment locally following this [online doc](https://github.com/microsoft/maro#install-maro-from-source):

1. Create your own conda env for MARO

2. Git clone the whole source code:

    ```sh
    git clone https://github.com/microsoft/maro.git
    ```

3. Run install maro script:

    ```sh
    bash scripts/install_maro.sh
    ```

    or

    ```powershell
    .\scripts\install_maro.bat
    ```

4. Set the PYTHONPATH before running:

    ```sh
    export PYTHONPATH=PATH-TO-MARO
    ```

    or

    ```powershell
    $Env:PYTHONPATH=PATH-TO-MARO
    ```

5. You can also quickly install the packages you may need by:

    ```sh
    pip install -r requirements.dev.txt
    pip install -r notebooks/requirements.nb.txt
    pip install -r examples/requirements.ex.txt
    ```

## Supply Chain Examples

### Topology Data

You can get the example *SCI* topologies with sampled data from SCI dataset (and also the data for topology *plant* and *super_vendor*) from the data blob by:

```sh
bash scripts/get_sci_data.sh
```

The scripts will automatically download below topologies to the *topologies* folder of supply chain:

- **SCI_10_default**: The stores only purchase products from the direct upstream storage warehouse.
- **SCI_10_cheapest_storage_enlarged**: The stores only purchase products from the one with cheapest transportation cost, it could be store or storage warehouse. Also, the storage capacity of stores and storage warehouses are enlarged to 10x.
- **SCI_10_shortest_storage_enlarged**: The stores only purchase products from the one with shortest leading time, it could be store or storage warehouse. Also, the storage capacity of stores and storage warehouses are enlarged to 10x.
- **SCI_500_default**: The stores only purchase products from the direct upstream storage warehouse.
- **SCI_500_cheapest_storage_enlarged**: The stores only purchase products from the one with cheapest transportation cost, it could be store or storage warehouse. Also, the storage capacity of stores and storage warehouses are enlarged to 10x.
- **SCI_500_shortest_storage_enlarged**: The stores only purchase products from the one with shortest leading time, it could be store or storage warehouse. Also, the storage capacity of stores and storage warehouses are enlarged to 10x.

### Simple random policy example

The simple random example shows the interface of the Supply Chain Simulator and illustrates how to interact with it. As you can see in line 68 of file [*examples/supply_chain/simple_random_example.py*](https://github.com/microsoft/maro/blob/sc_refinement/examples/supply_chain/simple_random_example.py#L68), we can deliver `ManufactureAction` and `ConsumerAction` to `Env`, and call function `step()` to trigger the simulation process. Try the simple example by:

```sh
python examples/supply_chain/simple_random_example.py
```

### Interaction with Non-RL policy

The complex example leverage the RL workflow in MARO. And the example code enable many configurations. Simpler configurations are listed in file [*examples/supply_chain/rl/config.py*](https://github.com/microsoft/maro/blob/sc_refinement/examples/supply_chain/rl/config.py). The basic ones you may need are:

- `ALGO` in line 48: The algorithm to use. "DQN" and "PPO" are RL algorithms, "EOQ" is a rule-based algorithm, "BSP" is an OR-algorithm base-stock policy.
- `TOPOLOGY` in line 67: The "plant" and "super_vendor" are toy topologies. You can use the "SCI(_XX)" ones if you add the topology under directory *maro/simulator/scenarios/supply_chain/topologies*
- `PLOT_RENDER` in line 72: Render figures to show important metrics during experiment or not.
- `EXP_NAME` in line 111: The experiment name, the experiment logs would be saved to the log path with `EXP_NAME` as the folder name.

With setting `ALGO = "EOQ"` in [line 48](https://github.com/microsoft/maro/blob/sc_refinement/examples/supply_chain/rl/config.py#L48), we can try to simulate with the rule-based policy. Since the non-rl policy does not require any training process, we can use *evaluate_only* mode by:

```sh
python examples/rl/run_rl_example.py examples/rl/supply_chain.yml --evaluate_only
```

### Interaction with RL policy

If you want to try trainable RL policy, you may also need to adjust the training workflow in file [*examples/rl/supply_chain.yml*](https://github.com/microsoft/maro/blob/sc_refinement/examples/rl/supply_chain.yml). The basic ones you may need are:

- `num_episodes` in line 15: Number of episode to run. Each episode is one cycle of roll-out and policy training.
- `eval_schedule` in line 17: Intervals between two evaluation process. `eval_schedule: 5` means will evaluate every 5 episodes.
- `interval` in line 31: Intervals between two dump action of policy network.

With setting `ALGO = "PPO"` in [line 48](https://github.com/microsoft/maro/blob/sc_refinement/examples/supply_chain/rl/config.py#L48) of *config.py*, we can try to simulate with the PPO algorithm based policy. The rl policy requires training process, so we need to enable training mode by:

```sh
python examples/rl/run_rl_example.py examples/rl/supply_chain.yml
```

### Much more complex configuration

The complex solution configurations are gathered in file [*examples/supply_chain/rl/rl_component_bundle.py*](https://github.com/microsoft/maro/blob/sc_refinement/examples/supply_chain/rl/rl_component_bundle.py), the ones you may concern about are:

- `get_agent2policy` in line 66: the mapping from the entity id in the scenario to the policy alias.
- `get_policy_creator` in line 84: what exactly the policy is for each policy alias.
- `get_trainer_creator` in line 97: the trainer for the policy training. It is related to what algorithm to use.
- `get_device_mapping` in line 109: the mapping from the policy alias to the training device.
- `get_policy_trainer_mapping` in line 135: the mapping from the policy alias to the trainer alias.

Besides, the **state shaping**, **action shaping** and **reward shaping** logics are defined in file [*examples/supply_chain/rl/env_sampler.py*](https://github.com/microsoft/maro/blob/sc_refinement/examples/supply_chain/rl/env_sampler.py), while [*examples/supply_chain/rl/rl_agent_state.py*](https://github.com/microsoft/maro/blob/sc_refinement/examples/supply_chain/rl/rl_agent_state.py) and [*examples/supply_chain/rl/or_agent_state.py*](https://github.com/microsoft/maro/blob/sc_refinement/examples/supply_chain/rl/or_agent_state.py) are used by **state shaping** logic.
