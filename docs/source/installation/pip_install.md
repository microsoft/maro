# Package

## Install from PyPI Using `pip`

```sh
pip install maro
```

## Install from Source

### Prerequisites

- [Python >= 3.6, < 3.8](https://www.python.org/downloads/)
- C++ Compiler
  - Linux or Mac OS X: `gcc`
  - Windows: [Build Tools for Visual Studio 2017](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15)

```sh
# If your environment is not clean, create a virtual environment first
python -m venv maro_venv
source maro_venv/bin/activate

# Install MARO from source, if you don't need CLI full feature
pip install -r ./maro/requirements.build.txt

# compile cython files
bash scripts/compile_cython.sh
pip install -e .

# Or with script
bash scripts/build_maro.sh
```
