# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Dict


class AbsExperiencePool(ABC):
    """Abstract class of experience pool"""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def put(self, category_data_batches: [Tuple[object, list]], align_check: bool = False) -> {object: [int]}:
        """Multi-category data put.

        Args:
            category_data_batches ([Tuple[object, list]]): Multi-category data tuple list,
                the 1st item of tuple is the category key,
                the 2nd item of tuple is the inserted items.
            align_check (bool): Enable align check for multi-category input items length.

        Returns:
            The new appended data indexes of each category.
        """
        pass

    @abstractmethod
    def update(self, category_data_batches: [Tuple[object, List[int], list]],
               align_check: bool = False) -> Dict[object, List[int]]:
        """Multi-category data update.

        Args:
            category_data_batches ([Tuple[object, [int], list]]): Multi-category data tuple list,
                the 1st item of tuple is the category key,
                the 2nd item of tuple is the updated item indexes,
                the 3rd item of tuple is the updated items.
            align_check (bool): Enable align check for multi-category input items and indexes length.

        Returns:
            Dict[object, List[int]]: The successfully updated data indexes of each category.
        """
        pass

    @abstractmethod
    def get(self, category_idx_batches: [Tuple[object, List[int]]], align_check: bool = False) -> Dict[object, List]:
        """Multi-category data get.

        Args:
            category_idx_batches ([Tuple[object, [int]]]): Multi-category index tuple list,
                the 1st item of tuple is the category key,
                the 2nd item of tuple is the got item indexes.
            align_check (bool): Enable align check for multi-category input indexes length.

        Returns:
            Dict[object, List]: The got data items of each category.
        """
        pass

    @abstractmethod
    def dump(self, path: str) -> bool:
        """
        Dump stores to disk.

        Args:
            path (str): Dumped path.
        Return:
            bool: Success or not.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bool:
        """
        Load stores from disk.

        Args:
            path (str): Loaded path.
        Returns:
            bool: Success or not.
        """
        pass

    @property
    @abstractmethod
    def size(self) -> Dict[object, int]:
        """
        Dict[object, int]: Size of each category
        """
        pass
