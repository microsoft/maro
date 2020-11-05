# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os

from requests import get
from tabulate import tabulate

if os.environ.get("REMOTE_DEBUG") == "on":
    port = os.environ.get("REMOTE_DEBUG_PORT")

    if not port:
        print("WARN: invalid port to enable remote debugging.")
    else:
        import ptvsd

        public_ip = get('https://api.ipify.org').text
        print("*******  Waiting for remote attach  ******")
        print(tabulate([['remote', public_ip, port], ['local', '127.0.0.1', port]], headers=['Host', 'IP', 'Port']))

        address = ('0.0.0.0', port)
        ptvsd.enable_attach(address)
        ptvsd.wait_for_attach()

        print("******  Attached  ******")
