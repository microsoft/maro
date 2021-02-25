import os

if __name__ == "__main__":
    print('starting runner...')
    os.system('cd tuner_metadata && python3 -m maro.automl.runner')
