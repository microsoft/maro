# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import os
import re
import subprocess
import sys
import unittest
from inspect import getmembers, isclass

test_file_re = re.compile(r'^test_.*.py$')



if __name__ == "__main__":

    script_folder = os.path.split(os.path.realpath(__file__))[0]

    # set working dir to tests folder
    os.chdir(script_folder)

    test_case_list=[]

    for path, _, file_names in os.walk("."):
        for fn in file_names:
            if test_file_re.match(fn):
                cur_script_path = os.path.join(path, fn)

                spliter = "\\" if sys.platform == "win32" else "/"
                
                module_name = ".".join(os.path.relpath(cur_script_path)[0:-3].split(spliter))

                test_case_list.append(module_name)
    
    print("loading test cases from following module")
    
    for i, n in enumerate(test_case_list):
        print(f"{i}: {n}")

    loader = unittest.TestLoader()

    suite = loader.loadTestsFromNames(test_case_list)

    runner = unittest.TextTestRunner()

    result = runner.run(suite)
