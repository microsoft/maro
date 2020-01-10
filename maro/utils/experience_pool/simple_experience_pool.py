# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from abc import ABC, abstractmethod
from collections import defaultdict
import random
import logging
import pickle
from typing import Callable, Tuple, List, Dict

from maro.utils.experience_pool.abs_experience_pool import AbsExperiencePool
from maro.utils.experience_pool.simple_store import SimpleStore

CategoryFilter = [Tuple[object, List[Callable[[Tuple[int, object]], Tuple[int, object]]]]]
CategorySampler = [Tuple[object, List[Tuple[Callable[[int, object], Tuple[int, object]], int]]]]


class SimpleExperiencePool(AbsExperiencePool):
    """Collection of the multi-category store, support cross-category batch operation on stores.
    """

    def __init__(self, size=None, replace=None):
        super(AbsExperiencePool, self).__init__()
        self._stores = defaultdict(lambda: SimpleStore(size=size, replace=replace))

    def put(self, category_data_batches: [Tuple[object, list]], align_check: bool = False) -> Dict[object, List[int]]:
        """Multi-category data put.

        Args:
            category_data_batches ([Tuple[object, list]]): Multi-category data tuple list,
                the 1st item of tuple is the category key,
                the 2nd item of tuple is the inserted items.
            align_check (bool): Enable align check for multi-category input items length.

        Returns:
            Dict[object, List[int]]: The new appended data indexes of each category.
        """
        if align_check and len(category_data_batches) > 1:
            first_items_len = len(category_data_batches[0][1])

            for i in range(1, len(category_data_batches)):
                assert (first_items_len == len(category_data_batches[i][1]))

        res, overwrite = {}, None
        for category, data in category_data_batches:
            fill, overwrite = self._stores[category].put(data, overwrite=overwrite)
            res[category] = (fill, overwrite)

        return res

    def update(self, category_data_batches: [Tuple[object, List[int], list]],
               align_check: bool = False) -> Dict[object, List[int]]:
        """Multi-category data update.

        Args:
            category_data_batches ([Tuple[object, List[int], list]]): Multi-category data tuple list,
                the 1st item of tuple is the category key,
                the 2nd item of tuple is the updated item indexes,
                the 3rd item of tuple is the updated items.
            align_check (bool): Enable align check for multi-category input items and indexes length.

        Returns:
            Dict[object, List[int]]: The successfully updated data indexes of each category.
        """
        if align_check and len(category_data_batches) > 1:
            first_idxs_len = len(category_data_batches[0][1])
            first_items_len = len(category_data_batches[0][2])
            assert (first_idxs_len == first_items_len)

            for i in range(1, len(category_data_batches)):
                assert (first_idxs_len == len(category_data_batches[i][1]) == len(
                    category_data_batches[i][2]))

        res = {}
        for cid in category_data_batches:
            res[cid[0]] = self._stores[cid[0]].update(cid[1], cid[2])

        return res

    def get(self, category_idx_batches: [Tuple[object, List[int]]], align_check: bool = False) -> Dict[object, List]:
        """Multi-category data get.

        Args:
            category_idx_batches ([Tuple[object, List[int]]]): Multi-category index tuple list,
                the 1st item of tuple is the category key,
                the 2nd item of tuple is the got item indexes.
            align_check (bool): Enable align check for multi-category input indexes length.

        Returns:
            Dict[object, List]: The got data items of each category.
        """

        if align_check and len(category_idx_batches) > 1:
            first_idxs_len = len(category_idx_batches[0][1])

            for i in range(1, len(category_idx_batches)):
                assert (first_idxs_len == len(category_idx_batches[i][1]))
        res = {}
        for ci in category_idx_batches:
            res[ci[0]] = self._stores[ci[0]].get(ci[1])

        return res

    def apply_multi_filters(self, category_filters: CategoryFilter, return_idx: bool = True) -> Dict[object, List]:
        """
        Multi-category batch filters.

        Args:
            category_filters ([Tuple[object, List[Callable[[Tuple[int, object]], Tuple[int, object]]]]]): Multi-category filter list. \n
                i.e. [('info1', [lambda tup: tup[1]['a'] == 1 and tup[1]['b'] == 1])] \n
                     [('info1', [lambda tup: tup[1]['a'] == 1, lambda tup: tup[1]['b'] == 1]), ('info2', [lambda tup: tup[1]['c'] == 1)]
            return_idx (bool): Return item index or item.
        Returns:
            Dict[object, List]: The final filtered indexes or items of each category.
        """
        res = {}
        for cf in category_filters:
            res[cf[0]] = self._stores[cf[0]].apply_multi_filters(
                filters=cf[1], return_idx=return_idx)

        return res

    def apply_multi_samplers(self, category_samplers: CategorySampler, return_idx: bool = True) -> Dict[object, List]:
        """Multi-category batch samplers.

        Args:
            category_samplers ([Tuple[object, List[Tuple[Callable[[int, object], Tuple[int, object]], int]]]]): Multi-category sampler list. \n
                i.e. [('info1', [(lambda i, o: (i, o['a']), 3), (lambda i, o: (i, o['b']), 10)])] \n
                     [('info1', [(lambda i, o: (i, o['a']), 3), (lambda i, o: (i, o['b']), 10)]), ('info2', [(lambda i, o: (i, o['c']), 3), (lambda i, o: (i, o['e']), 10)])]
            return_idx (bool): Return item index or item.
        Returns:
            Dict[object, List]: The final sampled indexes or items of each category.
        """
        res = {}
        for cs in category_samplers:
            res[cs[0]] = self._stores[cs[0]].apply_multi_samplers(
                samplers=cs[1], return_idx=return_idx)

        return res

    @property
    def size(self) -> Dict[object, int]:
        """Dict[object,int]: Each category store size.
        """
        res = {}
        for k in self._stores.keys():
            res[k] = self._stores[k].size
        return res

    def dump(self, path) -> bool:
        """Dump stores to disk.

        Args:
            path (str): Dumped path.
        Return:
            bool: Success or not.
        """
        try:
            with open(path, 'wb') as f:
                pickle.dump(self._stores, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception as e:
            logging.error(f'store dump error: {e}')
            return False

    def load(self, path) -> bool:
        """Load stores from disk.

        Args:
            path (str): Loaded path.
        Returns:
            bool: Success or not.
        """
        try:
            with open(path, 'rb') as f:
                self._stores = pickle.load(f)
            return True
        except Exception as e:
            logging.error(f'store load error: {e}')
            return False


if __name__ == '__main__':
    simple_experience_pool = SimpleExperiencePool()
    new_idxs = simple_experience_pool.put(
        [('reward', [1, 2, 3, 4, 5]), ('action', [0, 1, 0, 1, 0])])
    print(new_idxs)
