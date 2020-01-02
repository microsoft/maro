Single Host: Docker
=====================

Ubuntu
^^^^^^^

.. code-block:: bash

    # install pre-request
    sudo bash ./bin/install_pre_request.sh

    # build docker image
    gulp l/build_image

    # launch docker image
    gulp l/launch_container

    # login docker container
    gulp l/login_container

    # go to maro folder
    cd maro

    # run sample scenario
    python runner.py -u maro -e base_line
