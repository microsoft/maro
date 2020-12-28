# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import sys

from ..utils.subprocess import SubProcess

GET_PUBLIC_KEY_COMMAND = """\
ssh-keygen -t rsa -N '' -f ~/.ssh/id_rsa <<< y >/dev/null
cat ~/.ssh/id_rsa.pub
"""

if __name__ == "__main__":
    return_str = SubProcess.run(command=GET_PUBLIC_KEY_COMMAND)
    sys.stdout.write(return_str)
