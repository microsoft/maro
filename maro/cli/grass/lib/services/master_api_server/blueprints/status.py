# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import time

from flask import Blueprint

from ..jwt_wrapper import check_jwt_validity

# Flask related.

blueprint = Blueprint(name="status", import_name=__name__)
URL_PREFIX = "/v1/status"


@blueprint.route(f"{URL_PREFIX}", methods=["GET"])
@check_jwt_validity
def status():
    return {
        "status": "OK",
        "time": time.time()
    }
