import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_manufacture_unit_only(self):
        """
        Test if manufacture unit works as expect, like:

        1. with input sku
            . meet the storage limitation
            . not meet the storage limitation
            . with enough source sku
            . without enough source sku
            . with product rate
            . without product rate
        2. without input sku
            . meet the storage limitation
            . not meet the storage limitation
            . with product rate
            . without product rate

        """

        pass


if __name__ == '__main__':
    unittest.main()
