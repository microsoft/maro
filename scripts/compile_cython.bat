@ECHO OFF

chdir "%~dp0.."

pip install -r .\maro\requirements.build.txt

REM delete old .c files

DEL /F .\maro\backends\*.cpp

REM compile pyx into .c files
REM use numpy backend, and use a big memory block to hold array
cython .\maro\backends\backend.pyx .\maro\backends\np_backend.pyx .\maro\backends\raw_backend.pyx .\maro\backends\frame.pyx .\maro\data_lib\binary\binaryreader.pyx --cplus -3 -E NODES_MEMORY_LAYOUT=ONE_BLOCK -X embedsignature=True
