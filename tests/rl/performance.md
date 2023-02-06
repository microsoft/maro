# Performance for Gym Task Suite

We benchmarked the MARO RL Toolkit implementation in Gym task suite.
Some are compared to the benchmarks in [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/bench.html#).
Limited by the environment version difference<!-- and some others?-->,
there may be some gaps between the performance here and that in Spinning Up benchmarks.

The hyper-parameters are set to align with those used in [Spinning Up](https://spinningup.openai.com/en/latest/spinningup/bench.html#experiment-details):

- Network of on-policy algorithms: size (64, 32) with tanh units for both policy and value function;
- Network of off-policy algorithms: size (256, 256) with relu units;
- Batch size for on-policy algorithms: 4000 steps of interaction per batch update;
- Batch size for off-policy algorithms: size 100 for each gradient descent step;

## Walker2d

### Benchmark in Spinning Up - PyTorch Version

- Environment version: Walker2d-v3
- 3M timesteps

![Walker2d: PyTorch Version](https://spinningup.openai.com/en/latest/_images/pytorch_walker2d_performance.svg)

### Performance with MARO RL Toolkit

- Environment version: Walker2d-v4
- Training Mode: simple
- Rollout Mode: single
- Environment duration: 5000 ticks
- Num of episodes: 600
