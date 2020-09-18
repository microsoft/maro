
rem script to build wheel package on Windows
rem NOTE: Before building the wheels, please make sure you have setup-up the environment.
rem for python 3.6/3.7 we need vs++14

chdir "%~dp0.."

call scripts\compile_cython.bat

pip install -r maro/requirements.build.txt
pip install wheel
pip install --upgrade setuptools

python setup.py bdist_wheel