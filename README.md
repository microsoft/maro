![MARO LOGO](./docs/source/images/logo.svg)

MARO (Multi-Agent Resource Optimization) serves as a domain-specific RL solution,
which uses multi-agent RL to solve real-world resource optimization problems.
It can be applied to many important industrial domains,
such as empty container repositioning in logistics, bike repositioning in transportation,
VM provisioning in data center, assets management in finance, etc.
MARO has complete support on data processing, simulator building, RL algorithms selection, distributed training.

## Contents

| File/folder | Description                 |
| ----------- | --------------------------- |
| `maro`      | MARO source code.           |
| `examples`  | Showcase of MARO.           |
| `notebooks` | MARO quick-start notebooks. |

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

## Quick example

```python
from maro.simulator import Env

env = Env(scenario="ecr", topology="toy.5p_ssddd_l0.0", start_tick=0, durations=100)

metrics, decision_event, is_done = env.step(None)

while not is_done:
    metrics, decision_event, is_done = env.step(None)

print(f"environment metrics: {env.metrics}")

```

## Run playground

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