
rem script to build docker for playground image on Windows, this require the source code of maro

chdir "%~dp0.."

call .\scripts\compile_cython.bat

docker build -f ./docker_files/cpu.play.df . -t maro/playground:cpu