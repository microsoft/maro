from typing import Sequence

import numpy as np

from maro.rl import AbsStore


def get_item(data_dict, key_tuple):
    """Helper function to get the value in a hierarchical dictionary given the key path.

    Args:
        data_dict (dict): The data structure. For example:
            {
                "a": {
                    "b": 1,
                    "c": {
                        "d": 2,
                    }
                }
            }

        key_tuple (tuple): The key path to the target field. For example, given the data_dict above, the key_tuple
            ("a", "c", "d") should return 2.
    """
    for key in key_tuple:
        data_dict = data_dict[key]
    return data_dict


def set_item(data_dict, key_tuple, data):
    """The setter function corresponding to the get_item function."""
    for i, key in enumerate(key_tuple):
        if key not in data_dict:
            data_dict[key] = {}
        if i == len(key_tuple) - 1:
            data_dict[key] = data
        else:
            data_dict = data_dict[key]


class NumpyStore(AbsStore):
    def __init__(self, domain_type_dict, capacity):
        """
        Args:
            domain_type_dict (dict): The dictionary describing the name, structure and type of each field in the
                experience. Each field in the experience is the key-value pair in the folowing structure:
                (field_name): (size_of_an_instance, data_type, batch_first)

                For example:
                    ("s"): ((32, 64), np.float32, True)

                The field can be a hierarchical dictionary by identifying the full path to the root.

                For example:
                {
                    ("s", "p"): ((32, 64), np.float32, True)
                    ("s", "v"): ((48, ), np.float32, False),
                }
                Then the batch of experience returned by self.get(indexes) is:
                {
                    "s":
                        {
                            "p": numpy.array with size (batch, 32, 64),
                            "v": numpy.array with size (32, batch, 48),
                        }
                }
                Note that for the field ("s", "v"), the batch is in the 2nd dimension because the batch_first attribute
                is False.

            capacity (int): The maximum stored experience in the store.
        """
        super().__init__()
        self.domain_type_dict = dict(domain_type_dict)
        self.store = {
            key: np.zeros(
                shape=(capacity, *shape) if batch_first else (shape[0], capacity, *shape[1:]), dtype=data_type)
            for key, (shape, data_type, batch_first) in domain_type_dict.items()}
        self.batch_first_store = {key: batch_first for key, (_, _, batch_first) in domain_type_dict.items()}

        self.cnt = 0
        self.capacity = capacity

    def put(self, exp_dict: dict):
        """Insert a batch of experience into the store.

        If the store reaches the maximum capacity, this function will replace the experience in the store randomly.

        Args:
            exp_dict (dict): The dictionary of a batch of experience. For example:

                {
                    "s":
                        {
                            "p": numpy.array with size (batch, 32, 64),
                            "v": numpy.array with size (32, batch, 48),
                        }
                }

                The structure should be consistent with the structure defined in the __init__ function.

        Returns:
            indexes (numpy.array): The list of the indexes each experience in the batch is located in.
        """
        dlen = exp_dict["len"]
        append_end = min(max(self.capacity - self.cnt, 0), dlen)
        idxs = np.zeros(dlen, dtype=np.int)
        if append_end != 0:
            for key in self.domain_type_dict.keys():
                data = get_item(exp_dict, key)
                if self.batch_first_store[key]:
                    self.store[key][self.cnt: self.cnt + append_end] = data[0: append_end]
                else:
                    self.store[key][:, self.cnt: self.cnt + append_end] = data[:, 0: append_end]
            idxs[: append_end] = np.arange(self.cnt, self.cnt + append_end)
        if append_end < dlen:
            replace_idx = self._get_replace_idx(dlen - append_end)
            for key in self.domain_type_dict.keys():
                data = get_item(exp_dict, key)
                if self.batch_first_store[key]:
                    self.store[key][replace_idx] = data[append_end: dlen]
                else:
                    self.store[key][:, replace_idx] = data[:, append_end: dlen]
            idxs[append_end: dlen] = replace_idx
        self.cnt += dlen
        return idxs

    def _get_replace_idx(self, cnt):
        return np.random.randint(low=0, high=self.capacity, size=cnt)

    def get(self, indexes: np.array):
        """Get the experience indexed in the indexes list from the store.

        Args:
            indexes (np.array): A numpy array containing the indexes of a batch experience.

        Returns:
            data_dict (dict): the structure same as that defined in the __init__ function.
        """
        data_dict = {}
        for key in self.domain_type_dict.keys():
            if self.batch_first_store[key]:
                set_item(data_dict, key, self.store[key][indexes])
            else:
                set_item(data_dict, key, self.store[key][:, indexes])
        return data_dict

    def __len__(self):
        return min(self.capacity, self.cnt)

    def update(self, indexes: Sequence, contents: Sequence):
        raise NotImplementedError("NumpyStore does not support modifying the experience!")

    def sample(self, size, weights: Sequence, replace: bool = True):
        raise NotImplementedError("NumpyStore does not support sampling. Please use outer sampler to fetch samples!")

    def clear(self):
        """Remove all the experience in the store."""
        self.cnt = 0


class Shuffler:
    def __init__(self, store: NumpyStore, batch_size: int):
        """The helper class for fast batch sampling.

        Args:
            store (NumpyStore): The data source for sampling.
            batch_size (int): The size of a batch.
        """
        self._store = store
        self._shuffled_seq = np.arange(0, len(store))
        np.random.shuffle(self._shuffled_seq)
        self._start = 0
        self._batch_size = batch_size

    def next(self):
        """Uniformly sampling out a batch in the store."""
        if self._start >= len(self._store):
            return None
        end = min(self._start + self._batch_size, len(self._store))
        rst = self._store.get(self._shuffled_seq[self._start: end])
        self._start += self._batch_size
        return rst

    def has_next(self):
        """Check if any experience is not visited."""
        return self._start < len(self._store)
