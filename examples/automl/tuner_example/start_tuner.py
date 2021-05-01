import os

if __name__ == "__main__":
    file_path = os.path.abspath(__file__)
    os.chdir(os.path.dirname(file_path))
    os.system('cd tuner_metadata && python3 -m maro.automl.runner')
