# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import time

from flask import Flask

from .blueprints.jobs import blueprint as jobs_blueprint

# App related

app = Flask(__name__)
app.url_map.strict_slashes = False

# Blueprints related

app.register_blueprint(blueprint=jobs_blueprint)


@app.route("/status", methods=["GET"])
def status():
    return {
        "status": "OK",
        "time": time.time()
    }
