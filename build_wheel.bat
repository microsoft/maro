rem "Before building the wheels, please make sure you have setup-up the environment."
rem "for python 3.6/3.7 we need vs++14"

pip install -r maro/simulator/requirements.build.txt

python setup.py bdist_wheel