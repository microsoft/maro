@ECHO OFF

rem script to build maro locally on Windows, usually for development

chdir "%~dp0.."

rem compile cython files
call scripts\compile_cython.bat

python setup.py build_ext -i