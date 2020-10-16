[![License](https://img.shields.io/pypi/l/pymaro)](https://github.com/microsoft/maro/blob/master/LICENSE)
[![Platform](https://raw.githubusercontent.com/microsoft/maro/master/docs/source/images/badges/platform.svg)](https://pypi.org/project/pymaro/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pymaro.svg?logo=python&logoColor=white)](https://pypi.org/project/pymaro/#files)
[![Code Size](https://img.shields.io/github/languages/code-size/microsoft/maro)](https://github.com/microsoft/maro)
[![Docker Size](https://img.shields.io/docker/image-size/arthursjiang/maro)](https://hub.docker.com/repository/docker/arthursjiang/maro/tags?page=1)
[![Issues](https://img.shields.io/github/issues/microsoft/maro)](https://github.com/microsoft/maro/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/microsoft/maro)](https://github.com/microsoft/maro/pulls)
[![Dependencies](https://img.shields.io/librariesio/github/microsoft/maro)](https://libraries.io/pypi/pymaro)
[![test](https://github.com/microsoft/maro/workflows/test/badge.svg)](https://github.com/microsoft/maro/actions?query=workflow%3Atest)
[![build](https://github.com/microsoft/maro/workflows/build/badge.svg)](https://github.com/microsoft/maro/actions?query=workflow%3Abuild)
[![docker](https://github.com/microsoft/maro/workflows/docker/badge.svg)](https://hub.docker.com/repository/docker/arthursjiang/maro)
[![docs](https://readthedocs.org/projects/maro/badge/?version=latest)](https://maro.readthedocs.io/)
[![PypI Versions](https://img.shields.io/pypi/v/pymaro)](https://pypi.org/project/pymaro/#files)
[![Wheel](https://img.shields.io/pypi/wheel/pymaro)](https://pypi.org/project/pymaro/#files)
[![Citi Bike](https://raw.githubusercontent.com/microsoft/maro/master/docs/source/images/badges/citi_bike.svg)](https://maro.readthedocs.io/en/latest/scenarios/citi_bike.html)
[![CIM](https://raw.githubusercontent.com/microsoft/maro/master/docs/source/images/badges/cim.svg)](https://maro.readthedocs.io/en/latest/scenarios/container_inventory_management.html)
[![Gitter](https://img.shields.io/gitter/room/microsoft/maro)](https://gitter.im/Microsoft/MARO#)
[![Stack Overflow](https://raw.githubusercontent.com/microsoft/maro/master/docs/source/images/badges/stack_overflow.svg)](https://stackoverflow.com/questions/ask?tags=maro)
[![Releases](https://img.shields.io/github/release-date-pre/microsoft/maro)](https://github.com/microsoft/maro/releases)
[![Commits](https://img.shields.io/github/commits-since/microsoft/maro/latest/master)](https://github.com/microsoft/maro/commits/master)
[![Vulnerability Scan](https://github.com/microsoft/maro/workflows/vulnerability%20scan/badge.svg)](https://github.com/microsoft/maro/actions?query=workflow%3A%22vulnerability+scan%22)
[![Lint](https://github.com/microsoft/maro/workflows/lint/badge.svg)](https://github.com/microsoft/maro/actions?query=workflow%3Alint)
[![Coverage](https://img.shields.io/codecov/c/github/microsoft/maro)](https://codecov.io/gh/microsoft/maro)
[![Downloads](https://img.shields.io/pypi/dm/pymaro)](https://pypi.org/project/pymaro/#files)
[![Docker Pulls](https://img.shields.io/docker/pulls/arthursjiang/maro)](https://hub.docker.com/repository/docker/arthursjiang/maro)
[![Play with MARO](https://raw.githubusercontent.com/microsoft/maro/master/docs/source/images/badges/play_with_maro.svg)](https://hub.docker.com/r/arthursjiang/maro)

# [![MARO LOGO](./docs/source/images/logo.svg)](https://maro.readthedocs.io/en/latest/)

Multi-Agent Resource Optimization (MARO) platform is an instance of Reinforcement
learning as a Service (RaaS) for real-world resource optimization. It can be
applied to many important industrial domains, such as [container inventory
management](https://maro.readthedocs.io/en/v0.1/scenarios/container_inventory_management.html) 
in logistics, [bike repositioning](https://maro.readthedocs.io/en/v0.1/scenarios/citi_bike.html) 
in transportation, virtual machine provisioning in data centers, and asset management in finance. Besides
[Reinforcement Learning](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf) (RL),
it also supports other planning/decision mechanisms, such as
[Operations Research](https://en.wikipedia.org/wiki/Operations_research).

Key Components of MARO:

- Simulation toolkit: it provides some predefined scenarios, and the reusable
wheels for building new scenarios.
- RL toolkit: it provides a full-stack abstraction for RL, such as agent manager,
agent, RL algorithms, learner, actor, and various shapers.
- Distributed toolkit: it provides distributed communication components, interface
of user-defined functions for message auto-handling, cluster provision, and job orchestration.

![MARO Key Components](./docs/source/images/maro_overview.svg)

## Contents

| File/folder | Description                                                                                       |
| ----------- | ------------------------------------------------------------------------------------------------- |
| `maro`      | MARO source code.                                                                                 |
| `docs`      | MARO docs, it is host on [readthedocs](https://maro.readthedocs.io/en/latest/).                   |
| `examples`  | Showcase of MARO.                                                                                 |
| `notebooks` | MARO quick-start notebooks.                                                                       |

## Install MARO from [PyPI](https://pypi.org/project/pymaro/#files)

- Max OS / Linux

  ```sh
  pip install pymaro
  ```

- Windows

  ```powershell
  # Install torch first, if you don't have one.
  pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

  pip install pymaro
  ```

## Install MARO from Source ([Editable Mode](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs))

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

    ```powershell
    # If your environment is not clean, create a virtual environment firstly.
    python -m venv maro_venv
    .\maro_venv\Scripts\activate

    # You may need this for SecurityError in PowerShell.
    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted
    ```

- Install MARO

  - Mac OS / Linux

    ```sh
    # Install MARO from source.
    bash scripts/install_maro.sh
    ```

  - Windows

    ```powershell
    # Install MARO from source.
    .\scripts\install_maro.bat
    ```

## Quick Example

```python
from maro.simulator import Env

env = Env(scenario="cim", topology="toy.5p_ssddd_l0.0", start_tick=0, durations=100)

metrics, decision_event, is_done = env.step(None)

while not is_done:
    metrics, decision_event, is_done = env.step(None)

print(f"environment metrics: {env.metrics}")

```

## Run Playground

- Pull from [Docker Hub](https://hub.docker.com/repository/registry-1.docker.io/arthursjiang/maro/tags?page=1)

  ```sh
  # Run playground container.
  # Redis commander (GUI for redis) -> http://127.0.0.1:40009
  # Local host docs -> http://127.0.0.1:40010
  # Jupyter lab with maro -> http://127.0.0.1:40011
  docker run -p 40009:40009 -p 40010:40010 -p 40011:40011 arthursjiang/maro:cpu
  ```

- Build from source
  - Mac OS / Linux

    ```sh
    # Build playground image.
    bash ./scripts/build_playground.sh

    # Run playground container.
    # Redis commander (GUI for redis) -> http://127.0.0.1:40009
    # Local host docs -> http://127.0.0.1:40010
    # Jupyter lab with maro -> http://127.0.0.1:40011
    docker run -p 40009:40009 -p 40010:40010 -p 40011:40011 maro/playground:cpu
    ```

  - Windows

    ```powershell
    # Build playground image.
    .\scripts\build_playground.bat

    # Run playground container.
    # Redis commander (GUI for redis) -> http://127.0.0.1:40009
    # Local host docs -> http://127.0.0.1:40010
    # Jupyter lab with maro -> http://127.0.0.1:40011
    docker run -p 40009:40009 -p 40010:40010 -p 40011:40011 maro/playground:cpu
    ```

## Contributing

This project welcomes contributions and suggestions. Most contributions require
you to agree to a Contributor License Agreement (CLA) declaring that you have
the right to, and actually do, grant us the rights to use your contribution. For
details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether
you need to provide a CLA and decorate the PR appropriately (e.g., status check,
comment). Simply follow the instructions provided by the bot. You will only need
to do this once across all repos using our CLA.

This project has adopted the
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the
[Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com)
with any additional questions or comments.

## License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](./LICENSE) License.
