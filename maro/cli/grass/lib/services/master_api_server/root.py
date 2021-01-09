# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from flask import Flask

from .blueprints_v1.cluster import blueprint as cluster_blueprint
from .blueprints_v1.containers import blueprint as containers_blueprint
from .blueprints_v1.image_files import blueprint as image_files_blueprint
from .blueprints_v1.jobs import blueprint as jobs_blueprint
from .blueprints_v1.join_node_script import blueprint as join_node_script_blueprint
from .blueprints_v1.master import blueprint as master_blueprint
from .blueprints_v1.nodes import blueprint as nodes_blueprint
from .blueprints_v1.schedules import blueprint as schedules_blueprint
from .blueprints_v1.status import blueprint as status_blueprint

# App related

app = Flask(__name__)
app.url_map.strict_slashes = False

# Blueprints related

app.register_blueprint(blueprint=cluster_blueprint)
app.register_blueprint(blueprint=containers_blueprint)
app.register_blueprint(blueprint=image_files_blueprint)
app.register_blueprint(blueprint=jobs_blueprint)
app.register_blueprint(blueprint=join_node_script_blueprint)
app.register_blueprint(blueprint=master_blueprint)
app.register_blueprint(blueprint=nodes_blueprint)
app.register_blueprint(blueprint=schedules_blueprint)
app.register_blueprint(blueprint=status_blueprint)
