Single Host: OS X
==================

Install Prerequisites
---------------------

- Python 3.6

    - Recommend to install Anaconda 3

        - https://www.anaconda.com/download/#linux
        - Create new virtual Python 3.6 env via Anaconda:

        .. code-block:: bash

            $ conda -V  # Check conda is installed and in your PATH
            $ conda update conda  # Check conda is up to date
            $ conda create -n YOUR_ENV_NAME python=3.6 anaconda  # Create a python 3.6 virtual environment

        - Active virtual environment

        .. code-block:: bash

            $ source activate YOUR_ENV_NAME

    - OR directly install Python 3.6

        .. code-block:: bash

            sudo apt-get install python3.6
- GCC

    .. code-block:: bash

        sudo apt-get install gcc

- Gulp

    - Node.JS >= 8.0

        .. code-block:: bash

            $ curl -sL https://deb.nodesource.com/setup_8.x | sudo -E bash - && apt-get install -y nodejs

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

        $ bash ./build_maro.sh