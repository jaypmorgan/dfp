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
        out = dfp.lmap(lambda x: x[0], zip(range(10), range(20, 30)))
        self.assertEqual(out[0], 0)
        self.assertEqual(out[-1], 9)
        out = dfp.lmap(lambda x: x**2, range(10_000), parallel=True)
        self.assertEqual(out[0], 0)
        self.assertEqual(len(out), 10_000)
        self.assertEqual(out[-1], 9999**2)
        out = dfp.lmap(lambda x: x**2, map(lambda x: x, range(10_000)), parallel=True, progress=True)
        self.assertEqual(out[0], 0)
        self.assertEqual(len(out), 10_000)
        self.assertEqual(out[-1], 9999**2)
        out = dfp.lmap(lambda x: x**2, map(lambda x: x, range(10_000)), parallel=False, progress=True)
        self.assertEqual(out[0], 0)
        self.assertEqual(len(out), 10_000)
        self.assertEqual(out[-1], 9999**2)

    def test_lzip(self):
        out = dfp.lzip(dfp.tfilter(lambda i: i % 2 == 0, range(10)), dfp.tfilter(lambda i: i % 2 != 0, range(10)))
        self.assertEqual(out[0][0], 0)
        self.assertEqual(out[0][1], 1)
        self.assertEqual(out[1][0], 2)
        self.assertEqual(len(out), 5)

    def test_lfilter(self):
        out = dfp.lfilter(lambda i: i % 2 == 0, range(10))
        self.assertEqual(len(out), 5)
        self.assertEqual(out[0], 0)
        self.assertEqual(out[1], 2)
        out = dfp.lfilter(lambda i: i, [])
        self.assertEqual(len(out), 0)
        self.assertEqual(type(out), list)

    def test_first_rest(self):
        f, r = dfp.first_rest(range(10))
        self.assertEqual(f, 0)
        self.assertEqual(r[0], 1)
        self.assertEqual(len(r), 9)

    def test_nth(self):
        self.assertEqual(dfp.nth(range(10), 9), 9)

    def test_take(self):
        out = dfp.take(range(10), 5)
        self.assertEqual(dfp.first(out), 0)
        self.assertEqual(dfp.second(out), 1)

    def test_has_props(self):
        a = (('a', 1), ('b', 2),)
        self.assertTrue(dfp.has_props(a, {"a": 1}))
        self.assertFalse(dfp.has_props(a, {"b": 1}))
        self.assertTrue(dfp.has_props(a, {"a": [1, 2], "b": 2}))
        self.assertTrue(dfp.has_props(a, {"b": lambda v: v > 0}))

    def test_inverse(self):
        fn = lambda i: i == 0
        inv_fn = dfp.inverse(fn)
        out1 = dfp.lfilter(fn, range(10))
        out2 = dfp.lfilter(inv_fn, range(10))
        self.assertEqual(len(out1), 1)
        self.assertEqual(len(out2), 9)

    def test_pluck_item(self):
        record = (('a', 1), ('a', 2))
        self.assertEqual(dfp.pluck_item("a", record), 2)
        a = (('a', 1), ('b', 2),)
        self.assertEqual(dfp.pluck_item("b", a), 2)
        self.assertEqual(dfp.pluck_item("c", a), None)

    def test_pluck_list(self):
        a = ((('a', 1), ('b', 2),), (('a', 5), ('b', 4),),)
        out = dfp.pluck_list("a", a)
        self.assertEqual(out[0], 1)
        self.assertEqual(out[1], 5)
            
    def test_keys(self):
        record = (('a', 1), ('a', 2))
        k = dfp.keys(record)
        self.assertEqual(k, ('a',))
        record = (('a', 1), ('b', 2))
        k = dfp.keys(record)
        self.assertNotEqual(k, ('a',))
        self.assertEqual(k, ('a', 'b'))

    def test_join_string(self):
        self.assertEqual(dfp.join_strings(["foo", "bar"]), "foobar")
        self.assertEqual(dfp.join_strings(["foo", "bar"], "/"), "foo/bar")

    def test_join_paths(self):
        self.assertEqual(dfp.join_paths("/foo", "bar"), "/foo/bar")
        self.assertEqual(dfp.join_paths("/foo/bar//", "/baz"), "/foo/bar/baz")

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

    def test_port_lines(self):
        content = ["this is a test", "this is another line"]
        filepath = dfp.port_lines("/tmp/filename.txt", content)
        self.assertEqual(filepath, "/tmp/filename.txt")
        out_content = dfp.port_lines(filepath)
        self.assertEqual(out_content[0], "this is a test")
        self.assertEqual(out_content[1], "this is another line")

    def test_port_csv(self):
        content = [["this is a test", "this is another line"]]
        filepath = dfp.port_csv("/tmp/filename.csv", content)
        self.assertEqual(filepath, "/tmp/filename.csv")
        out_content = dfp.port_csv(filepath)
        self.assertEqual(out_content[0][0], "this is a test")
        self.assertEqual(out_content[0][1], "this is another line")

    def test_port_pickle(self):
        content = [["this is a test", "this is another line"]]
        filepath = dfp.port_pickle("/tmp/filename.pkl", content)
        self.assertEqual(filepath, "/tmp/filename.pkl")
        out_content = dfp.port_pickle(filepath)
        self.assertEqual(out_content[0][0], "this is a test")
        self.assertEqual(out_content[0][1], "this is another line")

    def test_port_json(self):
        content = [["this is a test", "this is another line"]]
        filepath = dfp.port_json("/tmp/filename.json", content)
        self.assertEqual(filepath, "/tmp/filename.json")
        out_content = dfp.port_json(filepath)
        self.assertEqual(out_content[0][0], "this is a test")
        self.assertEqual(out_content[0][1], "this is another line")
        
        

if __name__ == '__main__':
    unittest.main()
