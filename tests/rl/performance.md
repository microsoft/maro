# Performance for Gym Task Suite

We benchmarked the MARO RL Toolkit implementation in Gym task suite. Some are compared to the benchmarks in
[OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/bench.html#). We've tried to align the
hyper-parameters for these benchmarks , but limited by the environment version difference, there may be some gaps
between the performance here and that in Spinning Up benchmarks. Generally speaking, the performance is comparable.

## Experimental Setting

The hyper-parameters are set to align with those used in
[Spinning Up](https://spinningup.openai.com/en/latest/spinningup/bench.html#experiment-details):

**Batch Size**:

- For on-policy algorithms: 4000 steps of interaction per batch update;
- For off-policy algorithms: size 100 for each gradient descent step;

**Network**:

- For on-policy algorithms: size (64, 32) with tanh units for both policy and value function;
- For off-policy algorithms: size (256, 256) with relu units;

**Performance metric**:

- For on-policy algorithms: measured as the average trajectory return across the batch collected at each epoch;
- For off-policy algorithms: measured once every 10,000 steps by running the deterministic policy (or, in the case of SAC, the mean policy) without action noise for ten trajectories, and reporting the average return over those test trajectories;

**Total timesteps**: set to 4M for all task suites and algorithms.

More details about the parameters can be found in *tests/rl/tasks/*.

## Performance

Five environments from the MuJoCo Gym task suite are reported in Spinning Up, they are: HalfCheetah, Hopper, Walker2d,
Swimmer, and Ant. The commit id of the code used to conduct the experiments for MARO RL benchmarks is ee25ce1e97.
The commands used are:

```sh
# Step 1: Set up the MuJoCo Environment in file tests/rl/gym_wrapper/common.py

# Step 2: Use the command below to run experiment with ALGORITHM (ddpg, ppo, sac) and random seed SEED.
python tests/rl/run.py tests/rl/tasks/ALGORITHM/config.yml --seed SEED

# Step 3: Plot performance curves by environment with specific smooth window size WINDOWSIZE.
python tests/rl/plot.py --smooth WINDOWSIZE
```

|     **Env**     | **Spinning Up** | **MARO RL w/o Smooth** | **MARO RL w/ Smooth** |
|:---------------:|:---------------:|:----------------------:|:---------------------:|
| [**HalfCheetah**](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) | ![Hab](https://spinningup.openai.com/en/latest/_images/pytorch_halfcheetah_performance.svg) | ![Ha1](./log/HalfCheetah_1.png) | ![Ha11](./log/HalfCheetah_11.png) |
| [**Hopper**](https://gymnasium.farama.org/environments/mujoco/hopper/) | ![Hob](https://spinningup.openai.com/en/latest/_images/pytorch_hopper_performance.svg) | ![Ho1](./log/Hopper_1.png) | ![Ho11](./log/Hopper_11.png) |
| [**Walker2d**](https://gymnasium.farama.org/environments/mujoco/walker2d/) | ![Wab](https://spinningup.openai.com/en/latest/_images/pytorch_walker2d_performance.svg) | ![Wa1](./log/Walker2d_1.png) | ![Wa11](./log/Walker2d_11.png) |
| [**Swimmer**](https://gymnasium.farama.org/environments/mujoco/swimmer/) | ![Swb](https://spinningup.openai.com/en/latest/_images/pytorch_swimmer_performance.svg) | ![Sw1](./log/Swimmer_1.png) | ![Sw11](./log/Swimmer_11.png) |
| [**Ant**](https://gymnasium.farama.org/environments/mujoco/ant/) | ![Anb](https://spinningup.openai.com/en/latest/_images/pytorch_ant_performance.svg) | ![An1](./log/Ant_1.png) | ![An11](./log/Ant_11.png) |
