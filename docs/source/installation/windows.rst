Single Host: Windows
=====================

Install Prerequisites
---------------------

- Python 3.6

    - Recommend to install Anaconda 3

        - https://repo.anaconda.com/archive/Anaconda3-5.3.0-Windows-x86_64.exe
        - Create new virtual Python 3.6 env via Anaconda:

        .. code-block:: bash

            $ conda -V  # Check conda is installed and in your PATH
            $ conda update conda  # Check conda is up to date
            $ conda create -n YOUR_ENV_NAME python=3.6 anaconda  # Create a python 3.6 virtual environment

        - Active virtual environment

        .. code-block:: bash

            $ source activate YOUR_ENV_NAME

    - OR directly install Python 3.6

        - https://www.python.org/ftp/python/3.6.5/python-3.6.5-amd64.exe

    - Install required packages

        .. code-block:: bash

            $ pip install -r requirements.dev.txt

- C++ Build Tools

    - Download `Build Tools for Visual Studio 2017`

        - Direct download link https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15
        - OR go to https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017

    - Install

        - Windows 7 check `Visual C++ Build Tools`
        - Windows 10 check `Visual C++ Build Tools` or `Desktop Development With C++`

- Gulp

    - Node.JS >= 8.0

        https://nodejs.org/en/download/

    - Gulp 3.9.1

        .. code-block:: bash

            $ npm install --global gulp-cli
            $ npm install --save gulp@3.9.1

    - 3rd Party Packages

        .. code-block:: bash

            $ npm install

Build MARO
----------

    .. code-block:: bash

        $ python setup.py build_ext -i