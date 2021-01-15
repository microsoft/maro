import os

from bottle import route, run, static_file, request


ROOT_DIR = "/maro/files"

if not os.path.join(ROOT_DIR):
    os.mkdir(ROOT_DIR)

@route("/files/<filename:path>")
def send_static(filename):
    return static_file(filename, root=ROOT_DIR)


@route("/files/<experiment>/<episode:int>/<filename>", method="POST")
def upload_to_episode(experiment: str, episode:int, filename: str):
    experiment_path = os.path.join(ROOT_DIR, experiment)

    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    episode_path = os.path.join(experiment_path, str(episode))

    if not os.path.exists(episode_path):
        os.mkdir(episode_path)

    with open(os.path.join(episode_path, filename), mode="wb+") as fp:
        fp.write(request.body.read())

@route("/files/<experiment>/<filename>", method="POST")
def upload_to_experiment(experiment: str, filename: str):
    print(experiment, filename)


    #with open(os.path.join("./experiments/"))


run(host="localhost", port=9100)