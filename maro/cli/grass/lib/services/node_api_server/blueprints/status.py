# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import time

from flask import Blueprint

# Flask related.

blueprint = Blueprint(name="status", import_name=__name__)
URL_PREFIX = "/v1/status"


@blueprint.route(f"{URL_PREFIX}", methods=["GET"])
def status():
    return {
        "status": "OK",
        "time": time.time()
    }
