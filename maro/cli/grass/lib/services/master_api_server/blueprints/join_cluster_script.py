# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from flask import Blueprint, abort, send_from_directory

from ...utils.params import Paths

# Flask related.

blueprint = Blueprint(name="join_cluster_script", import_name=__name__)
URL_PREFIX = "/v1/joinClusterScript"


@blueprint.route(f"{URL_PREFIX}", methods=["GET"])
def get_init_node_script():
    try:
        return send_from_directory(
            directory=f"{Paths.ABS_MARO_SHARED}/lib/grass/scripts/node",
            filename="join_cluster.py",
            as_attachment=True
        )
    except FileNotFoundError:
        abort(404)
