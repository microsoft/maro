
Package
=======

Install MARO from `PyPI <https://pypi.org/project/pymaro/#files>`_
----------------------------------------------------------------------

* Max OS / Linux

  .. code-block:: sh

    pip install pymaro

* Windows

  .. code-block::

    # Install torch first, if you don't have one.
    pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

    pip install pymaro

Install MARO from Source (\ `Editable Mode <https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`_\ )
------------------------------------------------------------------------------------------------------------------------

* Prerequisites

  * `Python >= 3.7 <https://www.python.org/downloads/>`_
  * C++ Compiler

    * Linux or Mac OS X: ``gcc``
    * Windows: `Build Tools for Visual Studio 2017 <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15>`_

* Enable Virtual Environment

  * Mac OS / Linux

    .. code-block:: sh

       # If your environment is not clean, create a virtual environment firstly.
       python -m venv maro_venv
       source ./maro_venv/bin/activate

  * Windows

    .. code-block:: powershell

      # If your environment is not clean, create a virtual environment firstly.
      python -m venv maro_venv
      .\maro_venv\Scripts\activate

* Install MARO

  * Mac OS / Linux

    .. code-block:: sh

      # Install MARO from source.
      bash scripts/install_maro.sh

  * Windows

    .. code-block:: powershell

      # Install MARO from source.
      .\scripts\install_maro.bat
