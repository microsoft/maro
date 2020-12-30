# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import time

from flask import Flask

from .blueprints.containers import blueprint as container_blueprint

app = Flask(__name__)
app.url_map.strict_slashes = False

# Blueprints related

app.register_blueprint(blueprint=container_blueprint)


@app.route("/status", methods=["GET"])
def status():
    return {
        "status": "OK",
        "time": time.time()
    }
