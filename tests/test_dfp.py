import unittest

from src import dfp


class DFPTests(unittest.TestCase):

    def test_find(self):
        test_list = ['a', 'b', 'c', 'd']
        idx = dfp.find('b', test_list)
        self.assertEqual(idx, 1)
        idx = dfp.find('e', test_list)
        self.assertEqual(idx, None)
        idx = dfp.find([], test_list)
        self.assertEqual(idx, None)
        idx = dfp.find([], [])
        self.assertEqual(idx, None)
        idx = dfp.find([], ['a', 'b', [], 'd'])
        self.assertEqual(idx, 2)
        idx = dfp.find([], ['a', 'b', ['a'], 'd'])
        self.assertEqual(idx, None)
        idx = dfp.find(['a'], ['a', 'b', ['a'], 'd'])
        self.assertEqual(idx, 2)
        idx = dfp.find(['a  '], ['a', 'b', ['a'], 'd'])
        self.assertEqual(idx, None)

    def test_keys(self):
        record = (('a', 1), ('a', 2))
        k = dfp.keys(record)
        self.assertEqual(k, ('a',))
        record = (('a', 1), ('b', 2))
        k = dfp.keys(record)
        self.assertNotEqual(k, ('a',))
        self.assertEqual(k, ('a', 'b'))

    def test_take_batch(self):
        lst = list(range(6))
        out_lst = dfp.take_batch(lst, 0, 4)
        self.assertEqual(out_lst, [0, 1, 2, 3])
        self.assertEqual(dfp.take_batch(lst, 1, 4), [4, 5])
        lst = list()
        self.assertEqual(dfp.take_batch(lst, 5, 4), [])
        

if __name__ == '__main__':
    unittest.main()
