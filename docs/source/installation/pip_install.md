# Package

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

    ```powershell
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

    ```powershell
    # Install MARO from source.
    .\scripts\install_maro.bat
    ```
