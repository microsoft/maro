import multiprocessing

import numpy as np


def init_shared_memory(data_structure):
    """Initialize the data structure of the shared memory.

    Args:
        data_structure: The dictionary that describes the data structure. For example,
            {
                "a": (shape, type),
                "b": {
                        "b1": (shape, type),
                    }
            }
    """
    if isinstance(data_structure, tuple):
        mult = 1
        for i in data_structure[0]:
            mult *= i
        return multiprocessing.Array(data_structure[1], mult, lock=False)
    else:
        shared_data = {}
        for k, v in data_structure.items():
            shared_data[k] = init_shared_memory(v)
        return shared_data


def shared_data2numpy(shared_data, structure_info):
    if not isinstance(shared_data, dict):
        return np.frombuffer(shared_data, dtype=structure_info[1]).reshape(structure_info[0])
    else:
        numpy_dict = {}
        for k, v in shared_data.items():
            numpy_dict[k] = shared_data2numpy(v, structure_info[k])
        return numpy_dict


class SharedStructure:
    def __init__(self, data_structure):
        self.data_structure = data_structure
        self.shared = init_shared_memory(data_structure)

    def structuralize(self):
        return shared_data2numpy(self.shared, self.data_structure)
