# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from flask import Blueprint, send_from_directory, abort

# Flask related.

blueprint = Blueprint(name="join_node_script", import_name=__name__)
URL_PREFIX = "/v1/joinNodeScript"


@blueprint.route(f"{URL_PREFIX}", methods=["GET"])
def get_init_node_script():
    try:
        return send_from_directory(
            directory=os.path.expanduser("~/.maro/lib/grass/scripts/node"),
            filename="join_node.py",
            as_attachment=True
        )
    except FileNotFoundError:
        abort(404)
