python maro/utils/dashboard/package_data.py
# NOTE: currently we only support python3.6 and 3.7, need to be clearfy the python and packages version
# about manylinux: https://github.com/pypa/manylinux
docker run --rm -v "$PWD":/maro quay.io/pypa/manylinux2010_x86_64 bash /maro/build_wheel.sh