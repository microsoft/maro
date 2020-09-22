rem script to run all the test script under tests folder which the file name match test_xxxx.py

chdir "%~dp0.."

set "PYTHONPAH=."

call scripts/build_maro.bat

rem install requirements

pip install -r ./tests/requirements.test.txt

rem show coverage

coverage run --rcfile=./tests/.coveragerc

coverage report --rcfile=./tests/.coveragerc