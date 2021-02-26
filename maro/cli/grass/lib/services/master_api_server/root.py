# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


""" A Flask server for MARO Master API Server.

Hosted by gunicorn at systemd.
"""

from flask import Flask

from .blueprints.cluster import blueprint as cluster_blueprint
from .blueprints.containers import blueprint as containers_blueprint
from .blueprints.image_files import blueprint as image_files_blueprint
from .blueprints.jobs import blueprint as jobs_blueprint
from .blueprints.join_cluster_script import blueprint as join_cluster_script_blueprint
from .blueprints.master import blueprint as master_blueprint
from .blueprints.nodes import blueprint as nodes_blueprint
from .blueprints.schedules import blueprint as schedules_blueprint
from .blueprints.status import blueprint as status_blueprint
from .blueprints.visible import blueprint as visible_blueprint

# App related

app = Flask(__name__)
app.url_map.strict_slashes = False

# Blueprints related

app.register_blueprint(blueprint=cluster_blueprint)
app.register_blueprint(blueprint=containers_blueprint)
app.register_blueprint(blueprint=image_files_blueprint)
app.register_blueprint(blueprint=jobs_blueprint)
app.register_blueprint(blueprint=join_cluster_script_blueprint)
app.register_blueprint(blueprint=master_blueprint)
app.register_blueprint(blueprint=nodes_blueprint)
app.register_blueprint(blueprint=schedules_blueprint)
app.register_blueprint(blueprint=status_blueprint)
app.register_blueprint(blueprint=visible_blueprint)
