# temp file to quich launch mock runner
export PYTHONPATH="$PROJECT_ROOT;$PYTHONPATH"

# move to maro_sim folder to build the extensions
cd $PROJECT_ROOT

# build the cython extension first
python setup.py build_ext -i