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

    def test_for_each(self):
        out = []
        dfp.for_each(lambda name: out.append(name), ['a','b','c'])
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], 'a')
        self.assertEqual(out[-1], 'c')

    def test_lmap(self):
        out = dfp.lmap(lambda x: x, range(10))
        self.assertEqual(len(out), 10)
        self.assertEqual(out[0], 0)
        self.assertEqual(out[-1], 9)
        out = dfp.lmap(lambda x: x*2, range(10))
        self.assertEqual(len(out), 10)
        self.assertEqual(out[0], 0)
        self.assertEqual(out[-1], 18)
            
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

    def test_flatten_dict(self):
        dct = {"my_dict": {"foo": {"bar": {"baz": True}}}}
        flattened = dfp.flatten_dict(dct)
        self.assertEqual(flattened["my_dict.foo.bar.baz"], True)
        flattened = dfp.flatten_dict({})
        self.assertEqual(flattened, {})
        flattened = dfp.flatten_dict(dct, key_join_fn=lambda ki, kj: f"{ki}-{kj}")
        self.assertEqual(flattened["my_dict-foo-bar-baz"], True)
        flattened = dfp.flatten_dict(dct, key_join_fn=lambda ki, kj: kj)
        self.assertEqual(flattened["baz"], True)

    def test_merge_dicts(self):
        dicts = ({'test': 0}, {'test': 1}, {'test': 2})
        merged = dfp.merge_dicts(*dicts)
        self.assertEqual(len(merged['test']), 3)
        self.assertEqual(merged['test'][-1], 2)
        self.assertEqual(merged['test'][0], 0)
        self.assertEqual(dfp.merge_dicts({}), {})
        

if __name__ == '__main__':
    unittest.main()
