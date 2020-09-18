# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import subprocess
import sys

command = """\
ssh-keygen -t rsa -N '' -f ~/.ssh/id_rsa <<< y >/dev/null
cat ~/.ssh/id_rsa.pub
"""

if __name__ == "__main__":
    process = subprocess.Popen(command,
                               executable='/bin/bash',
                               shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8')
    stdout, stderr = process.communicate()
    if stderr:
        sys.stderr.write(stderr.strip('\n'))
    sys.stdout.write(stdout.strip('\n'))
