# Package

## Install from PyPI Using `pip`

```sh
pip install maro
```

## Install MARO from Source ([editable mode](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs))

- Prerequisites
  - [Python >= 3.6, < 3.8](https://www.python.org/downloads/)
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
