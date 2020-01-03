# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import tarfile
import os

cwd = os.getcwd()
path_filter = ('data','panels/line_chart/node_modules','panels/heatmap_chart/node_modules')
tar_rel_path = 'maro/utils/dashboard/dashboard_resource'

def tardir(path, tar_name, filter):
    if os.path.exists(os.path.join(cwd,tar_name)):
        os.remove(os.path.join(cwd,tar_name))
    with tarfile.open(tar_name, "w:gz") as tar_handle:
        for root, _, files in os.walk(path):
            if not os.path.relpath(root,  os.path.join(cwd, tar_rel_path)).startswith(filter):
                for file in files:
                    rel_path = os.path.relpath(root, os.path.join(cwd, tar_rel_path))
                    if rel_path == '.':
                        tar_handle.add(os.path.join(root, file), file)
                    else:
                        tar_handle.add(os.path.join(root, file), os.path.join(rel_path,file))
        tar_handle.close()

tar_abs_path = os.path.join(cwd, tar_rel_path)
if os.path.exists(tar_abs_path):
    tardir(tar_abs_path, 'maro/utils/dashboard/resource.tar.gz', path_filter)
else:
    print(f"{tar_abs_path} not found, please run under root directory of the maro project")
