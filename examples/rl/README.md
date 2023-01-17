# Reinforcement Learning (RL) Examples

This folder contains scenarios that employ reinforcement learning. MARO's RL toolkit provides scenario-agnostic workflows to run a variety of scenarios in single-thread, multi-process or distributed modes.

## How to Run

The entrance of a RL workflow is a YAML config file. For readers' convenience, we call this config file `config.yml` in the rest part of this doc. `config.yml` specifies the path of all necessary resources, definitions, and configurations to run the job. MARO provides a comprehensive template of the config file with detailed explanations (`maro/maro/rl/workflows/config/template.yml`). Meanwhile, MARO also provides several simple examples of `config.yml` under the current folder.

There are two ways to start the RL job:
- If you only need to have a quick look and try to start an out-of-box workflow, just run `python .\examples\rl\run_rl_example.py PATH_TO_CONFIG_YAML`. For example, `python .\examples\rl\run_rl_example.py .\examples\rl\cim.yml` will run the complete example RL training workflow of CIM scenario. If you only want to run the evaluation workflow, you could start the job with `--evaluate_only`.
- (**Require install MARO from source**) You could also start the job through MARO CLI. Use the command `maro local run [-c] path/to/your/config` to run in containerized (with `-c`) or non-containerized (without `-c`) environments. Similar, you could add `--evaluate_only` if you only need to run the evaluation workflow.

## Create Your Own Scenarios

You can create your own scenarios by supplying the necessary ingredients without worrying about putting them together in a workflow. It is necessary to create an ``__init__.py`` under your scenario folder (so that it can be treated as a package) and expose a `rl_component_bundle_cls` interface. The MARO's RL workflow will use this interface to create a `RLComponentBundle` instance and start the RL workflow based on it. a `RLComponentBundle` instance defines all necessary components to run a RL job. You can go through the doc string of `RLComponentBundle` for detailed explanation, or just read one of the examples to learn its basic usage.

## Example

For a complete example, please check `examples/cim/rl`.
