

cd "$(dirname $(readlink -f $0))/.."

pip install -r ./maro/requirements.build.txt

# delete old .c files
rm -f ./maro/backends/*.c

# compile pyx into .c files
# use numpy backend, and use a big memory block to hold array
cython ./maro/backends/backend.pyx ./maro/backends/np_backend.pyx ./maro/backends/raw_backend.pyx ./maro/backends/frame.pyx -3 -E FRAME_BACKEND=NUMPY,NODES_MEMORY_LAYOUT=ONE_BLOCK -X embedsignature=True
