@ECHO OFF

rem Script to install MARO in editable mode on Windows,
rem usually for development.

chdir "%~dp0.."

rem Install dependencies.
pip install -r .\maro\requirements.build.txt

rem Compile cython files.
call .\scripts\compile_cython.bat

call .\scripts\install_torch.bat

rem Install MARO in editable mode.
pip install -e .