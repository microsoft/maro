# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import unittest
from maro.utils.dashboard import DashboardBase
import sys
import random

dashboard = DashboardBase('Hello_World', '.')


class TestDashboardMethods(unittest.TestCase):

    def test_ranklist(self):
        # Test case 01: upoload to ranklist
        dashboard.upload_to_ranklist(ranklist = 'test_shortage_ranklist', fields={
                                     'shortage': random.random()*100})

    def test_hello_world(self):
        # Test case 02: upoload to hello world
        for i in range(10):
            fields = {'student_01': random.random(
            )*10*i, 'student_02': random.random()*15*i}
            tag = {'ep': i}
            measurement = 'score'
            dashboard.send(fields=fields, tag=tag, measurement=measurement)


if __name__ == '__main__':
    unittest.main()
