# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


""" A Flask server for MARO Node API Server.

Hosted by gunicorn at systemd.
"""

from flask import Flask

from .blueprints.containers import blueprint as container_blueprint
from .blueprints.status import blueprint as status_blueprint

app = Flask(__name__)
app.url_map.strict_slashes = False

# Blueprints related

app.register_blueprint(blueprint=container_blueprint)
app.register_blueprint(blueprint=status_blueprint)
