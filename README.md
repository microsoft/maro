![MARO LOGO](./docs/source/images/logo.svg)

Multi-Agent Resource Optimization (MARO) platform is an instance of Reinforcement learning as a Service (RaaS) for real-world resource optimization.
It can be applied to many important industrial domains,
such as container inventory management in logistics, bike repositioning in
transportation, virtual machine provisioning in data centers, and asset
management in finance. Besides [Reinforcement Learning](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf) (RL), it also supports other planning/decision mechanisms, such as
[Operations Research](https://en.wikipedia.org/wiki/Operations_research).

Key Components of MARO:

- Simulation toolkit: it provides some predefined scenarios, and the reusable wheels for building new scenarios.
- RL toolkit: it provides a full-stack abstraction for RL, such as agent manager, agent, RL algorithms, learner, actor, and various shapers.
- Distributed toolkit: it provides distributed communication components, interface of user-defined functions for message auto-handling, cluster provision, and job orchestration.

![MARO Key Components](./docs/source/images/maro_overview.svg)

## Contents

| File/folder | Description                                                                                       |
| ----------- | ------------------------------------------------------------------------------------------------- |
| `maro`      | MARO source code.                                                                                 |
| `docs`      | MARO docs, it is host on [readthedocs](https://maro.readthedocs.io/en/latest/index.html#).        |
| `examples`  | Showcase of MARO.                                                                                 |
| `notebooks` | MARO quick-start notebooks.                                                                       |

## Prerequisites

- [Python == 3.6/3.7](https://www.python.org/downloads/)

## Install MARO from PyPI

```sh
pip install maro
```

## Install MARO from Source ([editable mode](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs))

- Prerequisites
  - C++ Compiler
    - Linux or Mac OS X: `gcc`
    - Windows: [Build Tools for Visual Studio 2017](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15) 

- Enable Virtual Environment
  - Mac OS / Linux

    ```sh
    # If your environment is not clean, create a virtual environment firstly.
    python -m venv maro_venv
    source ./maro_venv/bin/activate
    ```

  - Windows

    ```ps
    # If your environment is not clean, create a virtual environment firstly.
    python -m venv maro_venv
    .\maro_venv\Scripts\activate
    ```

- Install MARO

  - Mac OS / Linux

    ```sh
    # Install MARO from source.
    bash scripts/install_maro.sh
    ```

  - Windows

    ```ps
    # Install MARO from source.
    .\scripts\install_maro.bat
    ```

## Quick Example

```python
from maro.simulator import Env

env = Env(scenario="ecr", topology="toy.5p_ssddd_l0.0", start_tick=0, durations=100)

metrics, decision_event, is_done = env.step(None)

while not is_done:
    metrics, decision_event, is_done = env.step(None)

print(f"environment metrics: {env.metrics}")

```

## Run Playground

```sh
# Build playground image
docker build -f ./docker_files/cpu.play.df . -t maro/playground:cpu

# Run playground container
# Redis commander (GUI for redis) -> http://127.0.0.1:40009
# Local host docs -> http://127.0.0.1:40010
# Jupyter lab with maro -> http://127.0.0.1:40011
docker run -p 40009:40009 -p 40010:40010 -p 40011:40011 maro/playground:cpu
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](./LICENSE) License.