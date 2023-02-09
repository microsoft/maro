# Performance for Gym Task Suite

We benchmarked the MARO RL Toolkit implementation in Gym task suite.
Some are compared to the benchmarks in [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/bench.html#).
Limited by the environment version difference<!-- and some others?-->,
there may be some gaps between the performance here and that in Spinning Up benchmarks.

## Experimental Setting

The hyper-parameters are set to align with those used in [Spinning Up](https://spinningup.openai.com/en/latest/spinningup/bench.html#experiment-details):

**Batch Size**:

- For on-policy algorithms: 4000 steps of interaction per batch update;
- For off-policy algorithms: size 100 for each gradient descent step;

**Network**:

- For on-policy algorithms: size (64, 32) with tanh units for both policy and value function;
- For off-policy algorithms: size (256, 256) with relu units;

**Performance metric**:

- For on-policy algorithms: measured as the average trajectory return across the batch collected at each epoch;
- For off-policy algorithms: measured once every 10,000 steps by running the deterministic policy (or, in the case of SAC, the mean policy) without action noise for ten trajectories, and reporting the average return over those test trajectories;

**Total timesteps**: set to 3M for all task suites and algorithms.

## Benchmark in Spinning Up - PyTorch Version

Five environments from the MuJoCo Gym task suite are reported in Spinning Up, they are: HalfCheetah, Hopper, Walker2d, Swimmer, and Ant.

**HalfCheetah-v3**:

The introduction of this task can be found [here](https://gymnasium.farama.org/environments/mujoco/half_cheetah/).

![HalfCheetah: PyTorch Version](https://spinningup.openai.com/en/latest/_images/pytorch_halfcheetah_performance.svg)

**Hopper-v3**:

The introduction of this task can be found [here](https://gymnasium.farama.org/environments/mujoco/hopper/).

![Hooper: PyTorch Version](https://spinningup.openai.com/en/latest/_images/pytorch_hopper_performance.svg)

**Walker2d-v3**:

The introduction of this task can be found [here](https://gymnasium.farama.org/environments/mujoco/walker2d/).

![Walker2d: PyTorch Version](https://spinningup.openai.com/en/latest/_images/pytorch_walker2d_performance.svg)

**Swimmer-v3**:

The introduction of this task can be found [here](https://gymnasium.farama.org/environments/mujoco/swimmer/).

![Swimmer-v3: PyTorch Version](https://spinningup.openai.com/en/latest/_images/pytorch_swimmer_performance.svg)

**Ant-v3**:

The introduction of this task can be found [here](https://gymnasium.farama.org/environments/mujoco/ant/).

![Ant-v3: PyTorch Version](https://spinningup.openai.com/en/latest/_images/pytorch_ant_performance.svg)

## Performance with MARO RL Toolkit

| **Algorithm** |     **Env**     | **1.5M** | **1.5M/baseline** | **3M** | **3M/baseline** | **gap in 10%?** |
|:-------------:|:---------------:|:--------:|:-----------------:|:------:|:---------------:|:---------------:|
|    **PPO**    | **HalfCheetah** |          |            ~ 2.5K |        |            ~ 4K |                 |
|               |    **Hopper**   |          |            ~ 2.3K |        |          ~ 2.5K |                 |
|               |   **Walker2d**  |          |            ~ 1.5K |        |          ~ 2.6K |                 |
|               |   **Swimmer**   |          |             ~ 113 |        |           ~ 118 |                 |
|               |     **Ant**     |          |            ~ 1.2K |        |            ~ 3K |                 |
|    **SAC**    | **HalfCheetah** |          |             ~ 12K |        |         ~ 13.7K |                 |
|               |    **Hopper**   |          |            ~ 3.5K |        |          ~ 3.5K |                 |
|               |   **Walker2d**  |          |            ~ 4.7K |        |            ~ 5K |                 |
|               |   **Swimmer**   |          |              ~ 43 |        |            ~ 43 |                 |
|               |     **Ant**     |          |           ~ 4.05K |        |            ~ 5K |                 |
