# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import unittest
from maro.utils.dashboard import DashboardBase
import sys
import random
import time

test_name = f'Hello_World_{time.time()}'

dashboard = DashboardBase(experiment=test_name, log_folder='.')


class TestDashboardMethods(unittest.TestCase):

    def test_ranklist(self):
        # Test case 01: upoload to ranklist
        fields = {'shortage': random.random()*100}
        dashboard.upload_to_ranklist(ranklist='test_shortage_ranklist', fields=fields)
        results = dashboard._connection.client.query('select * from "test_shortage_ranklist" where "experiment"= \'{test_name}\'')
        points = results.get_points()
        for point in points:
            assert(point['shortage'] == fields['shortage'])

    def test_hello_world(self):
        # Test case 02: upoload to hello world
        org_data = {}
        for i in range(10):
            fields = {'student_01': random.random()*10*i, 'student_02': random.random()*15*i}
            tag = {'ep': i}
            org_data[i] = fields
            measurement = 'score'
            dashboard.send(fields=fields, tag=tag, measurement=measurement)
        results = dashboard._connection.client.query('select * from "score" where "experiment"= \'{test_name}\'')
        points = results.get_points()
        for point in points:
            assert(point['student_01'] == org_data[point['ep']]['student_01'])
            assert(point['student_02'] == org_data[point['ep']]['student_02'])


if __name__ == '__main__':
    unittest.main()
