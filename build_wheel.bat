rem "Before building the wheels, please make sure you have setup-up the environment."
rem "for python 3.6/3.7 we need vs++14"
python maro/utils/dashboard/package_data.py

pip install -r maro/simulator/requirements.build.txt

python setup.py bdist_wheel