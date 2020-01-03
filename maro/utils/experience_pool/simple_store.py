# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import random
from typing import Callable, Tuple, List


class SimpleStore(object):
    """Simple store, support basic data batch operations: put, get, update, filter, sampling.
    """

    def __init__(self):
        self._internal_list = []

    def put(self, items: list) -> List[int]:
        """Put new items.

        Args:
            items (list): Item object list.

        Returns:
            List[int]: The new appended item indexes.
        """
        pre_size = self.size
        self._internal_list.extend(items)
        post_size = self.size
        return [i for i in range(pre_size, post_size)]

    def get(self, idxs: [int]) -> list:
        """Get items.

        Args:
            idxs (int): Item indexes list.

        Returns:
            list: The got item list.
        """
        return [self._internal_list[i] for i in idxs]

    def update(self, idxs: [int], items: list) -> List[int]:
        """Update selected items.

        Args:
            idxs ([int]): Item indexes list.
            items (list): Item list, which has the same length as idxs.

        Returns:
            List[int]: The updated item index list.
        """
        assert(len(idxs) == len(items))

        res = []

        for idx, item in zip(idxs, items):
            if idx < self.size:
                self._internal_list[idx] = item
                res.append(idx)

        return res

    def clear(self):
        """Clear current store.
        """
        self._internal_list = []

    def apply_multi_filters(self, filters: [Callable[[Tuple[int, object]], Tuple[int, object]]], return_idx: bool = True) -> list:
        """Multi-filter method.
            The next layer filter input is the last layer filter output.

        Args:
            filters ([Callable[[Tuple[int, object]], Tuple[int, object]]]): Filter list, each item is a lambda function. \n
                The lambda input is a tuple, (index, object). \n
                i.e. [lambda tup: tup[1]['a'] == 1 and tup[1]['b'] == 1]
            return_idx (bool): Return filtered indexes or items.

        Returns:
            list: Filtered indexes or items. i.e. [1, 2, 3], ['a', 'b', 'c']
        """
        tuples = enumerate(self._internal_list)
        for f in filters:
            tuples = filter(f, tuples)

        if return_idx:
            idxs = [t[0] for t in tuples]
            return idxs
        else:
            objs = [t[1] for t in tuples]
            return objs

    def apply_multi_samplers(self, samplers: [Tuple[Callable[[int, object], Tuple[int, object]], int]], return_idx: bool = True) -> list:
        """Multi-samplers method.
            The next layer sampler input is the last layer sampler output.

        Args:
            samplers ([Tuple[Callable[[int, object], Tuple[int, object]], int]]): Sampler list, each sampler is a tuple. \n
                The 1st item of the tuple is a lambda function. \n
                    The 1st lambda input is index, the 2nd lambda input is a object. \n
                The 2nd item of the tuple is the sample size. \n
                i.e. [(lambda i, o: (i, o['a']), 3)]
            return_idx (bool): Return sampled indexes or items.

        Returns:
            list: Sampled indexes or items.i.e. [1, 2, 3], ['a', 'b', 'c']
        """
        idxs = range(self.size)
        objs = self._internal_list
        for sampler in samplers:
            tuples = map(sampler[0], idxs, objs)
            x_tuples = tuples
            idxs, weights = zip(*[(t[0], t[1]) for t in tuples])
            sampled_idxs = random.choices(idxs, weights=weights, k=sampler[1])
            idxs = [i for i in sampled_idxs]
            objs = [self._internal_list[i] for i in sampled_idxs]

        if return_idx:
            return idxs
        else:
            objs = [self._internal_list[i] for i in idxs]
            return objs

    @property
    def size(self):
        """Current store size.
        """
        return len(self._internal_list)


if __name__ == '__main__':
    store = SimpleStore()
    new_idxs = store.put(
        [{'a': 1, 'b': 2}, {'a': 2, 'b': 3}, {'a': 1, 'b': 1}])
    print('new idxs:', new_idxs)
    new_idxs = store.put(
        [{'a': 1, 'b': 9}, {'a': 2, 'b': 5}, {'a': 1, 'b': 8}])
    print('new idxs:', new_idxs)
    filtered_idxs = store.apply_multi_filters(
        filters=[lambda tup: tup[1]['a'] == 1, lambda tup: tup[1]['b'] == 1])
    print('filtered idxs:', filtered_idxs)
    print('get filtered objs', store.get(filtered_idxs))
    filtered_objs = store.apply_multi_filters(
        filters=[lambda tup: tup[1]['a'] == 1 and tup[1]['b'] == 1], return_idx=False)
    print('filtered objs:', filtered_objs)

    sampled_idxs = store.apply_multi_samplers(
        samplers=[(lambda i, o: (i, o['a']), 3), (lambda i, o: (i, o['b']), 10)])
    print('sampled idxs:', sampled_idxs)
    print('get sampled objs:', store.get(sampled_idxs))
    sampled_objs = store.apply_multi_samplers(samplers=[(lambda i, o: (
        i, o['a']), 3), (lambda i, o: (i, o['b']), 10)], return_idx=False)
    print('sampled objs:', sampled_objs)

    print('get all items:', store.get(range(store.size)))
    store.update([1], [{'c': 100}])
    print('get all items after update:', store.get(range(store.size)))

    print('current size:', store.size)
    store.clear()
    print('size after clean:', store.size)
