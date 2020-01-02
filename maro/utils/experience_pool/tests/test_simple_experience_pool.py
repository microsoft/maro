# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import numpy as np
import os
import random
import time
import unittest

from utils.experience_pool.simple_experience_pool import SimpleExperiencePool


class TestSimpleExperiencePoolMethods(unittest.TestCase):
    def setUp(self):
        self.simple_experience_pool = SimpleExperiencePool()

    def test_put(self):
        idx_res = self.simple_experience_pool.put(
            [('reward', [1, 2, 3, 4, 5]), ('action', [0, 1, 0, 1, 0])])
        self.assertEqual(idx_res['reward'], [0, 1, 2, 3, 4])
        size_res = self.simple_experience_pool.size
        self.assertEqual(size_res['reward'], 5)
        self.assertEqual(size_res['action'], 5)

    def test_get(self):
        self.simple_experience_pool.put(
            [('reward', [1, 2, 3, 4, 5]), ('action', [0, 1, 0, 1, 0])])
        item_res = self.simple_experience_pool.get([('reward', [0, 1])])
        self.assertListEqual(item_res['reward'], [1, 2])

    def test_update(self):
        idx_res = self.simple_experience_pool.put(
            [('reward', [1, 2, 3, 4, 5]), ('action', [0, 1, 0, 1, 0])])
        self.simple_experience_pool.update([('reward', [1], [{'a': 1}])])
        item_res = self.simple_experience_pool.get([('reward', [1])])
        self.assertDictEqual(item_res['reward'][0], {'a': 1})

    def test_filter(self):
        self.simple_experience_pool.put([('info', [{'a': 5}, {'a': 2}, {'a': 3}, {
                                        'a': 4}, {'a': 5}]), ('action', [0, 1, 0, 1, 0])])
        idx_res = self.simple_experience_pool.apply_multi_filters(
            category_filters=[('info', [lambda tup: tup[1]['a'] == 5])])
        self.assertListEqual(idx_res['info'], [0, 4])

        item_res = self.simple_experience_pool.apply_multi_filters(
            category_filters=[('info', [lambda tup: tup[1]['a'] == 5])], return_idx=False)
        self.assertDictEqual(
            item_res, self.simple_experience_pool.get([('info', [0, 4])]))

    def test_sampler(self):
        self.simple_experience_pool.put([('info', [{'a': 1}, {'a': 0}, {'a': 0}, {
                                        'a': 0}, {'a': 0}]), ('action', [0, 1, 0, 1, 0])])
        idx_res = self.simple_experience_pool.apply_multi_samplers(
            category_samplers=[('info', [(lambda i, o: (i, o['a']), 3)])])
        self.assertListEqual(idx_res['info'], [0, 0, 0])

        item_res = self.simple_experience_pool.apply_multi_samplers(
            category_samplers=[('info', [(lambda i, o: (i, o['a']), 3)])], return_idx=False)
        self.assertDictEqual(
            item_res, self.simple_experience_pool.get([('info', [0, 0, 0])]))

    @unittest.skipIf(os.environ.get('TEST_SPEED') != 'on',
                     'Do not require speed test, set environment variable TEST_SPEED=on for opening.')
    def test_speed(self):
        states = []
        infos = []
        item_num = 1000000
        state_dim = (100, 10)
        init_start = time.time()
        for i in range(item_num):
            states.append(np.random.rand(*state_dim))
            infos.append({'a': random.randint(1, 100),
                          'b': random.randint(1, 100)})
        init_end = time.time()
        print(
            f'init {item_num} items time cost: {(init_end - init_start) * 1000} ms')

        # put 100k items, less than 500 ms
        put_start = time.time()
        self.simple_experience_pool.put([('state', states), ('info', infos)])
        put_end = time.time()
        self.assertLessEqual((put_end - put_start) * 1000, 500)
        print(
            f'put {item_num} items time cost: {(put_end - put_start) * 1000} ms')

        # get 100k items, less than 500 ms
        get_start = time.time()
        self.simple_experience_pool.get([('state', range(item_num))])
        get_end = time.time()
        self.assertLessEqual((get_end - get_start) * 1000, 500)
        print(
            f'get {item_num} items time cost: {(get_end - get_start) * 1000} ms')

        # update 100k items, less than 1 s
        new_states = states[::-1]
        update_start = time.time()
        self.simple_experience_pool.update(
            [('state', range(item_num), new_states)])
        update_end = time.time()
        self.assertLessEqual((update_end - update_start) * 1000, 1000)
        print(
            f'update {item_num} items time cost: {(update_end - update_start) * 1000} ms')

        # filter on 100k items, less than 500ms
        filter_start = time.time()
        self.simple_experience_pool.apply_multi_filters(
            [('info', [lambda tup: tup[1]['a'] == 5,
                       lambda tup: tup[1]['b'] == 1])])
        filter_end = time.time()
        self.assertLessEqual((filter_end - filter_start) * 1000, 500)
        print(
            f'filter {item_num} items time cost: {(filter_end - filter_start) * 1000} ms')

        # sample on 100k items, less than 2s
        sample_start = time.time()
        self.simple_experience_pool.apply_multi_samplers([('info', [(lambda i, o: (i, o['a']), 1024),
                                                                    (lambda i, o: (i, o['b']), 512)])])
        sample_end = time.time()
        self.assertLessEqual((sample_end - sample_start) * 1000, 2000)
        print(
            f'sample {item_num} items time cost: {(sample_end - sample_start) * 1000} ms')


if __name__ == '__main__':
    unittest.main()
