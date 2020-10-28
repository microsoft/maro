

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    cd "$(dirname $(readlink -f $0))/.."
elif [[ "$OSTYPE" == "darwin"* ]]; then
    cd "$(cd "$(dirname "$0")"; pwd -P)/.."
fi

pip install -r ./maro/requirements.build.txt

# delete old .c files
rm -f ./maro/backends/*.cpp

# compile pyx into .c files
# use numpy backend, and use a big memory block to hold array
cython ./maro/backends/backend.pyx ./maro/backends/np_backend.pyx ./maro/backends/raw_backend.pyx ./maro/backends/frame.pyx --cplus -3 -E NODES_MEMORY_LAYOUT=ONE_BLOCK -X embedsignature=True
