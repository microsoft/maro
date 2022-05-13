
rem script to build docker for playground image on Windows, this require the source code of maro

chdir "%~dp0.."

call .\scripts\compile_cython.bat

docker build -f ./docker_files/cpu.playground.df . -t maro2020/playground
