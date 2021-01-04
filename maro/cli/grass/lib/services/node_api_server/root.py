# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from flask import Flask

from .blueprints_v1.containers import blueprint as container_blueprint
from .blueprints_v1.status import blueprint as status_blueprint

app = Flask(__name__)
app.url_map.strict_slashes = False

# Blueprints related

app.register_blueprint(blueprint=container_blueprint)
app.register_blueprint(blueprint=status_blueprint)
