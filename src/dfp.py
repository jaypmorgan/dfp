"""A suite of functions to promote functional programming in Python.

DFP, or Dogmatic Functional Procedures, is a library of functions to
make functional programming in Python easier. In this way, DFP
contains many functions to build useful abstractions through function
compositions and transformations over data. Many of these functions
are built with generic data types in mind, and not specifically
related to astrophyical data.

"""
import os
import re
import math
import types
from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable, Optional, Iterator, Union
from collections.abc import Iterable
from itertools import product, chain
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import namedtuple

import pandas as pd
from tqdm.asyncio import tqdm


Record = tuple[str, Any]
Records = Union[dict[str, Any], tuple[Record]]


@dataclass
class Sequence(Iterable):
    """
    Abstraction of an Iterable that supports map, filter, and reduce.
    The intention of this class is to provide an alternative
    description of applying functional program operations from the
    perspective of an object. For example, we can create an
    instantiation of Sequence passing some data-structure during the
    invocation of the class:

    >>> from src.dfp import Sequence
    >>> seq = Sequence(range(100))

    This will create a new Sequence object, to which we may then apply
    operations, such as doubling the value of each element in the
    sequence:

    >>> seq.map(lambda x: x * 2)

    The return of this `.map()` call is another Sequence with each
    element being doubled.

    This is the alternative of using the `lmap` function, passing both
    the function and the data-structure to map over:

    >>> from src.dfp import lmap
    >>> lmap(lambda x: x * 2, range(100))

    One benefit of the Sequence class is that chaining operations,
    depending on your perspective, may look cleaner than the function
    based approach. For instance, let's only double the even numbers,
    removing any odd elements before:

    >>> (seq
    >>>  .filter(lambda x: x % 2 == 0)
    >>>  .map(lambda x: x * 2))

    The function only approach to this same problem may be to use
    `pipe` or `compose`:

    >>> from src.dfp import pipe, lfilter
    >>> pipe(
    >>>     range(100),
    >>>     lambda seq: lfilter(lambda x: x % 2 == 0, seq),
    >>>     lambda seq: lmap(lambda x: x * 2))

    In this latter example, we must 'wrap' our `lfilter` and `lmap` to
    prevent Python from immediately calling these functions.

    The design of DFP is flexible and based on the user's
    preference. If you want to use Sequence, there is no comprise as
    the class's methods resolve to the functions such as lmap/tmap
    etc.

    Methods
    -------
    filter:
        Filter a sequence.
    map:
        Apply a function to each element.
    reduce:
        Apply the reduce operation to the sequence.
    """

    data: Iterable

    def __iter__(self) -> Iterator:
        return self.data.__iter__()

    def __str__(self):
        return "Sequence(" + str(self.data) + ")"

    def __getitem__(self, item):
        return self.data[item]

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.data)

    def __next__(self):
        return self.data.__next__()

    def __eq__(self, other):
        """Are the elements of this sequence equal to another sequence
        or iterable?

        :param other: The other sequence or iterable to compare against.
        :returns: True or False

        >>> from src.dfp import Sequence
        >>> Sequence(range(10)) == Sequence(range(10))
        True

        However, if one sequence is longer than the other:
        >>> Sequence(range(10)) == Sequence(range(11))
        False

        As we're comparing elements, the `other` needn't be of type Sequence:

        >>> Sequence(range(5)) == [0, 1, 2, 3, 4]
        True
        """
        return (
            len(self) == len(other)
            and
            all(lmap(lambda arg: arg[0] == arg[1], lzip(self.data, other))))

    def map(self, f, parallel: bool = False, p_workers: int = 4):
        """Apply function `f` to every element of the sequence
        resulting in a new sequence of elements

        :param f: The function to apply to each element.
        :param parallel: (default: False) Whether to run the map in parallel
            threads.
        :param p_workers: (default: 4) The number of paraellel threads.
        :type f: Callable
        :type parallel: Bool
        :type p_workers: int
        :returns: A new sequence with each element being `f(x)`.

        >>> from src.dfp import Sequence
        >>> Sequence(range(5)).map(lambda x: x * 2)
        Sequence((0, 2, 4, 6, 8))
        """
        return Sequence(tmap(f, self.data, parallel, p_workers))

    def filter(self, f):
        """Filter a sequence of elements where each element is only
        included in the new sequence if `f(x) == True`.

        :param f: The boolean function to apply to each element that determines
            if the element should be included in the resulting sequence.
        :type f: Callable.
        :returns: A new sequence with elements only included if `f(x) == True`

        >>> from src.dfp import Sequence
        >>> Sequence(range(10)).filter(lambda x: x % 2 == 0)
        Sequence((0, 2, 4, 6, 8))
        """
        return Sequence(tfilter(f, self.data))

    def reduce(self, f, init: Any = 0):
        """Apply a reduction operation to the sequence. If the result
        not a singleton, a new Sequence is returned, else the
        singleton is returned.

        :param f: The reduction function to apply to each element
             of the sequence.
        :param init: (default: 0) The initial value.
        :type f: Callable
        :type init: Any
        :returns: A new sequence if the result is not a singleton.

        >>> from src.dfp import Sequence
        >>> Sequence(range(10)).reduce(lambda t, s: t + s)
        45
        >>> Sequence(range(10)).reduce(lambda t, s: t+[s] if s%2==0 else t, [])
        Sequence([0, 2, 4, 6, 8])

        Reduction operations don't actually have to 'reduce'
        anything. It can also expand the sequence. For example, in
        instance we're constructing a new sequence where each element
        is included twice, therefore, the resulting sequence is twice
        the original sequence.

        >>> Sequence(range(10)).reduce(lambda t, s: t + [s, s], [])
        Sequence([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
        """
        res = lreduce(f, self.data, init)
        if isinstance(res, Iterable):
            return Sequence(res)
        return res


def for_each(f, lst):
    """Apply `f` to each element of `lst`

    `for_each` abstracts a simple for loop where a function `f` is
    applied to each element of `lst`. `for_each` doesn't return
    anything but `f` can be used to add the result of `f` to a list
    within the scope of the caller.

    Parameters
    ----------
    f : Callable
        The function to call/apply to each element of `lst`.
    lst : Iterable
        The iterable list/tuple/ etc that `f` should be applied to.

    Examples
    --------
    FIXME: Add docs.

    """
    for el in lst:
        f(el)


def itemise(lst, idx_name: str = "idx", val_name: str = "val"):
    """Named-tuple enumeration"""
    pack = namedtuple("pack", [idx_name, val_name])
    idx = 0
    for item in lst:
        yield pack(idx, item)
        idx += 1


def itemize(lst, idx_name: str = "idx", val_name: str = "val"):
    """Named-tuple enumeration"""
    return itemise(lst, idx_name, val_name)


def lmap(
    f: Callable,
    lst: Iterable,
    parallel: bool = False,
    p_workers: int = 4,
    progress: bool = False,
    progress_fn: Callable = tqdm,
    p_type: str = "thread",
) -> list:
    """Apply function `f` to each element of `lst`. Return the results
    as a list.

    :param f: The function to apply to each element.
    :param lst: The iterable of elements to apply the function to.
    :param parallel: Boolean (false by default) flag that specifies if
        the function is applied in parallel using multiple threads. For
        very short functions/iterables this is slower. However, if your
        function is dependant on io, then you can get very fast speed ups
        with this. Results are still in order.
    :param p_workers: The number of workers to run in parallel.
    :type f: Callable
    :type lst: Iterable
    :type parallel: boolean
    :type p_workers: int
    :returns: A list with `f` applied to each element of `lst`.

    >>> lmap(lambda x: x*2, range(5))
    [0, 2, 4, 6, 8]
    """
    return list(tmap(f, lst, parallel, p_workers, progress, progress_fn, p_type))


def tmap(
        f: Callable,
        lst: Iterable,
        parallel: bool = False,
        p_workers: int = 4,
        progress: bool = False,
        progress_fn: Callable = tqdm,
        p_type: str = "thread",
) -> tuple:
    """Apply function `f` to each element of `lst`. Return the results
    as a list.

    :param f: The function to apply to each element.
    :param lst: The iterable of elements to apply the function to.
    :param parallel: Boolean (false by default) flag that specifies if
        the function is applied in parallel using multiple threads. For
        very short functions/iterables this is slower. However, if your
        function is dependent on io, then you can get very fast speed-ups
        with this. Results are still in order.
    :param p_workers: The number of workers to run in parallel.
    :param progress_fn: The function to create a progress bar.
    :param p_type: The type of multi-processing to use (i.e. thread or
        process).
    :type f: Callable
    :type lst: Iterable
    :type parallel: boolean
    :type p_workers: int
    :type progress_fn: Callable
    :type p_type: str
    :returns: A tuple with `f` applied to each element of `lst`.

    >>> tmap(lambda x: x*2, range(5))
    (0, 2, 4, 6, 8)
    >>> tmap(lambda x: x*2, range(5), parallel=True)
    (0, 2, 4, 6, 8)

    With `tmap`, the user can enable a progress par by using the argument
    `argument=True`.

    >>> tmap(lambda x: x*2, range(5), progress=True)

    This progress bar will also handle asynchronous operations, and
    therefore it is perfectly legal to use both `progress` and
    `parallel` at the same time. Note: due to the un-predictability of
    asynchronous function calls, the estimated time to complete
    reported by the progress bar will not be exactly accurate.

    If you want to customise the progress bar, you can pass a progress
    bar to the `progress_fn` argument. This also allows you to use a
    progress bar other than `tqdm`. This argument expects a callable
    function with a single argument -- the data to iterate
    over. Typical usage of this argument is with partial:

    >>> from functools import partial
    >>> from tqdm.auto import tqdm
    >>> tmap(lambda x: x * 2, range(5), progress=True,
             progress_fn=partial(tqdm, desc="Multiplying numbers by 2"))

    When this progress bar appears, it will have the correct
    description.

    """
    pbar = identity if not progress else progress_fn
    if isinstance(lst, (types.GeneratorType, map)):
        lst = list(lst)
    ProcessPool = ThreadPoolExecutor if p_type == "thread" else ProcessPoolExecutor
    if parallel:
        with ProcessPool(max_workers=p_workers) as executor:
            if progress:
                result = tuple(pbar(executor.map(f, lst), total=len(lst)))
            else:
                result = tuple(executor.map(f, lst))
    else:
        result = tuple(map(f, pbar(lst)))
    return result


def tzip(*lst) -> tuple:
    """Eagerly zip iterables returning a tuple.

    :param lst: The iterables to zip together.
    :type lst: Iterable
    :returns: Tuple of zipped elements.

    >>> from src.dfp import tzip
    >>> tzip(['a', 'b'], [1, 2])
    (('a', 1), ('b', 2))
    """
    return tuple(zip(*lst))


def lzip(*lst):
    """Eagerly zip iterables returning a list.

    :param lst: The iterables to zip together.
    :type lst: Iterable
    :returns: List of zipped elements.

    >>> from src.dfp import lzip
    >>> lzip(['a', 'b'], [1, 2])
    [('a', 1), ('b', 2)]
    """
    return list(zip(*lst))


def tfilter(f, lst):
    """Eagerly filter a list returning a tuple of elements where
    `f(x)` returns True.

    :param f: The function that returns True/False, True the element
         is included in the result.
    :param lst: The iterable to filter.
    :type f: Callable
    :type lst: Iterable
    :returns: Tuple of elements where `f(x)` is True

    >>> from src.dfp import tfilter
    >>> tfilter(lambda x: x % 2 == 0, range(10))
    (0, 2, 4, 6, 8)
    """
    return tuple(filter(f, lst))


def lfilter(f, lst):
    """Eagerly filter a list returning a list of elements where `f(x)`
    is True.

    :param f: The function to apply to each element to test if
         should be included in the list.
    :param lst: The iterable to filter.
    :type f: Callable
    :type lst: Iterable
    :returns: A list of elements where `f(x)` is True

    >>> from src.dfp import lfilter
    >>> lfilter(lambda x: x % 2 == 0, range(10))
    [0, 2, 4, 6, 8]
    """
    return list(filter(f, lst))


def treduce(f, lst, init):
    """Eagerly apply a reduce operation, returning a tuple if the
    result is not a singleton.

    :param f: The reduction function `f(x, y) -> z`, e.g.
         lambda sum, element: sum + element
    :param lst: The iterable to reduce.
    :param init: The initial value before applying `f` for the first time.
    :returns: A singleton element or a tuple of elements as defined
         by how `f` reduces the iterable.

    >>> from src.dfp import treduce
    >>> treduce(lambda sum, element: sum + element, range(10), 0)
    45

    Reduce operations are very general in that its possible to
    reimplement `map` and `filter` methods.

    >>> treduce(lambda lst, el: lst + [el] if el%2==0 else lst, range(10), [])
    (0, 2, 4, 6, 8)
    """
    result = reduce(f, lst, init)
    if isinstance(result, Iterable):
        return tuple(result)
    return result


def lreduce(f, lst, init):
    """Eagerly apply a reduce operation, returning a list if the
    result is not a singleton.

    :param f: The reduction function `f(x, y) -> z`, e.g.
        lambda sum, element: sum + element
    :param lst: The iterable to reduce.
    :param init: The initial value before applying `f` for the first time.
    :returns: A singleton element or a list of elements as defined by
        how `f` reduces the iterable.

    >>> from src.dfp import lreduce
    >>> lreduce(lambda sum, element: sum + element, range(10), 0)
    45

    Reduce operations are very general in that its possible to
    re-implement `map` and `filter` methods.

    >>> lreduce(lambda lst, el: lst + [el] if el%2==0 else lst, range(10), [])
    [0, 2, 4, 6, 8]
    """
    result = reduce(f, lst, init)
    if isinstance(result, Iterable):
        return list(result)
    return result


def first(lst):
    """Return the first element of an iterable.

    :param lst: The iterable to index into.
    :returns: The first element.

    >>> from src.dfp import first
    >>> first([0, 1, 2])
    0
    """
    return lst[0]


def second(lst):
    """Return the second element of an iterable.

    :param lst: The iterable to index into.
    :returns: The second element of the iterable

    >>> from src.dfp import second
    >>> second([0, 1, 2])
    1
    """
    return lst[1]


def rest(lst):
    """Return the iterable excluding the first element.

    :param lst: The iterable to index into.
    :returns: The iterable without the first element.

    >>> from src.dfp import rest
    >>> rest([0, 1, 2])
    [1, 2]
    """
    return lst[1:]


def first_rest(lst) -> tuple:
    """Deconstruct the first and rest of an iterable in one statement.

    :param lst: The iterable to deconstruct.
    :returns: A tuple with the first element being the first element
        of the list and the second element is the rest of the
        iterable.

    >>> from src.dfp import first_rest
    >>> first_rest([0, 1, 2])
    (0, [1, 2])

    This makes it more simple when assigning to variables than passing
    the same iterable to both `first` and `rest` functions.

    >>> my_list = [0, 1, 2]
    >>> f, r = first_rest(my_list)
    >>> f
    0
    >>> r
    [1, 2]

    """
    return first(lst), rest(lst)


def nth(lst, n):
    """Return the `n`-th element (using 0-based indexing) of an
    iterable.

    :param lst: The iterable to retrieve the element from.
    :param n: The index of the element to retrieve.
    :type lst: Iterable
    :type n: int
    :returns: The `n`-th element of `lst`.

    >>> nth(['a', 'b', 'c'], 2)
    'c'
    """
    return lst[n]


def slice(lst, start=None, stop=None, step=1):
    """Slice an iterable by a start/stop/step index.

    :param lst: The iterable to index into.
    :param start: The start index, default is the start of the iterable.
    :param stop: The end index, default is the end of the iterable.
    :param step: How many indexes to step by, default is 1.

    >>> from src.dfp import slice
    >>> slice(list(range(10)), 1)
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> slice(list(range(10)), stop=5)
    [0, 1, 2, 3, 4]
    >>> slice(list(range(10)), start=2, stop=9, step=2)
    [2, 4, 6, 8]
    """
    return lst[start:stop:step]


def unique(lst, key: Optional[str] = None, how=lambda l, v: v not in l):
    """
    Return the unique elements using `key` as the key to determine
    unique elements. If multiple elements with the same value exist,
    the latter in the sequence will be returned.
    """
    if key is None:
        # here we assume that we're just looking for unique elements in an iterable, i.e. not a record.
        return tuple(reversed(
            treduce(lambda t, x: t + [x] if x not in t else t, reversed(lst), [])))
    else:
        return tuple(reversed(
            treduce(lambda t, x: t + [x] if how(pluck_list(key, t), pluck_item(key, x)) else t,
                        reversed(lst), [])))


def take(lst, n):
    """Take a number of elements from a sequence"""
    return slice(lst, start=0, stop=n)


def take_subset(lst: Iterable, indexes: Iterable, bools: bool = False) -> tuple:
    """Take `indexes` from `lst`.

    Take elements from `lst` using `indexes` to denote which elements
    to take. This is equivalent to [lst[idx] for idx in indexes],
    where the result a subset of `lst`.

    Parameters
    ----------
    lst : Iterable
        The iterable to take a subset from.
    indexes : Iterable
        An iterable of indexes.
    bools : bool
        Optional argument to specify if indexes consists of a list of
        indexes (bools=False) or if indexes is a list of boolean
        values where True denotes that element x_i should be included
        in the subset.

    Examples
    --------
    >>> from src.dfp import take_subset
    >>> lst = ['a', 'b', 'c']
    >>> take_subset(lst, [0, 1])
    ('a', 'b')

    From this example, we create a list named `a`, with the elements
    'a', 'b', and 'c'. Then we take a subset of this list using the
    indexes 0, and 1.

    We can also specify boolean values as indexes. For this the size
    of `lst` and `bools` must be the same size. If a boolean value at
    index i is True, then element at index i in lst will be returned
    in the subset. For example:

    >>> indexes = [True, False, True]
    >>> assert len(indexes) == len(lst)
    >>> take_subset(lst, indexes, bools=True)
    ('a', 'c')

    Returns
    -------
    tuple
        A tuple representing the subset of `lst`.

    """
    if bools:
        indexes = treduce(
            lambda t, x: t + [x.idx] if x.val else t, itemise(indexes), [])
    return treduce(lambda t, idx: t + [nth(lst, idx[1])],
                   enumerate(indexes), [])


def picknmix(*iterables):
    """Take elements from iterables one at a time

    :param iterables: The iterables to sample from.
    :type iterables: Iterable[Iterable]
    :returns: A tuple of successive elements from each iterable in turn.

    >>> picknmix([0, 2], [1, 3])
    (0, 1, 2, 3)

    `picknmix` returns successive elements to the smallest iterable:
    >>> picknmix([0, 2], [1])
    [0, 1]
    >>> picknmix((0, 2), (1, 3))
    (0, 1, 2, 3)
    """
    iterables = list(iterables)
    n_elems = reduce(
        lambda t, s: len(s) if len(s) < t else t, iterables, math.inf)
    num_iterables = len(iterables)
    iterable_type = type(first(iterables))
    return iterable_type(
        tmap(
            lambda idx: iterables[idx[1]][idx[0]],
            product(range(n_elems), range(num_iterables))))


def last(lst):
    """Get the last element of an iterable

    :param lst: The iterable to index over.
    :type lst: Iterable
    :returns: The last item of the iterable.
    """
    return nth(lst, -1)


def but_last(lst):
    """Get all elements except the last one"""
    return slice(lst, 0, -1)


def identity(x):
    """The identity function, return the input.

    :param x: The value to return.
    :type x: Any
    :returns: The input.
    """
    return x


def none_fn(*args, **kwargs):
    """Always return None"""
    del args
    del kwargs
    return None


def join_string(fields, delim=", "):
    val = first(fields)
    fields = rest(fields)
    if len(fields) > 0:
        res = join_string(fields, delim)
        return val + delim + res
    return val


#####################################################################################
#                                   Set joins                                       #
#####################################################################################


def _check_join_keys(keys):
    return [keys]*2 if isinstance(keys, str) else keys  # duplicate keys to left and right


def spread(records, key_col, val_col):
    """Spread a table"""
    return tmap(
        lambda row: treduce(
            lambda t, item: t + [item] if first(item) != key_col and first(item) != val_col else t,
            row,
            [(pluck_item(key_col, row), pluck_item(val_col, row))]),
        records)


def join_left(left, right, by):
    """Join records by left"""
    by = _check_join_keys(by)
    return tmap(
        lambda l: reduce(
            lambda l, row: reduce(
                lambda l, item: add(
                    l,
                    first(item) if first(item) not in keys(l) else first(item) + "_y",
                    second(item)),
                row, l),
            tfilter(lambda r: pluck_item(by[1], r) == pluck_item(by[0], l), right),
            l),
        left)


def join_right(left, right, by):
    """Join records by right"""
    by = _check_join_keys(by)
    return join_left(right, left, by=list(reversed(by)))


def join_inner(left, right, by):
    """Join records by inner"""
    by = _check_join_keys(by)
    k = set(pluck_list(by[1], right))
    left = tfilter(lambda l: pluck_item(by[0], l) in k, left)
    return join_left(left, right, by)


def join_full(left, right, by):
    pass


def join(left, right, by: Union[str, list[str]], how: str = "inner"):
    """Join records"""
    fn = {
        "inner": join_inner,
        "full": join_full,
        "left": join_left,
        "right": join_right}.get(how)
    if fn:
        return fn(left, right, by)
    raise ValueError(f"Unknown join operation: {how}")


def join_paths(*args) -> str:
    """Join paths together.

    Parameters
    ----------
    *args : Path, str
        All the paths you wish to join together.

    Returns
    -------
    str
        A string representation of the joined path.

    Examples
    --------
    >>> from src.dfp import join_paths
    >>> join_paths("/path/to", "something")
    '/path/to/something'

    The every argument, except the last is assumed to be a folder. For example:

    >>> join_paths("path/to", "something", "else")
    'path/to/something/else'

    This function also works with pathlib.Path values, but always returns a `str`.

    >>> from pathlib import Path
    >>> join_paths(Path("/path/"), "//to/something")
    '/path/to/something'

    Here we also see the advantage of using `join_paths` where the resulting
    path is always clean, in that it doesn't have duplicate '/' values.

    """
    return re.sub(r"/{2,}", "/", join_string(lmap(str, args), "/"))


def orderedset(x):
    """Ordered set"""
    return treduce(lambda t, s: t + [s] if s not in t else t, x, [])


def alloc(value, length) -> list[Any]:
    """Create a list with values of specified length

    :param value: The value of every value in the new list.
    :param length: The length of the allocated list.
    :type value: Any
    :type length: int

    >>> alloc(0, 5)
    [0, 0, 0, 0, 0]
    >>> alloc(["?"], 5)
    [['?'], ['?'], ['?'], ['?'], ['?']]
    >>> alloc(alloc(0, 5), 5)
    [[0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]]
    """
    return [value] * length


def flatten(lst):
    """Flatten a nested list into a single list"""
    return reduce(
        lambda t, x: t+[x] if not isinstance(x, list) else t+flatten(x),
        lst, [])


def un_nest(iterable):
    return tuple(chain(*iterable))


def group_by(key, iterable):
    """Group records by key"""
    groups = orderedset(pluck_list(key, iterable))
    return tmap(
        lambda group: tfilter(
            lambda item: pluck_item(key, item) == group, iterable),
        groups), groups


def find(token, inlist) -> Optional[int]:
    """Return the first index where token is found in `inlist`. If the
    token is not found, None is returned.

    """
    return reduce(
        lambda t, x: first(x) if second(x) == token and t is None else t,
        enumerate(inlist), None)


def member(token, in_list):
    """
    Returns the rest of a iterable for which token is found. For example,

    >>> member('s', ['a', 's', 'b'])
    ['s', 'b']

    If the token is not found, an empty iterable is returned
    """
    start_token = find(token, in_list)
    return slice(
        in_list, start_token if start_token is not None else len(in_list))


def filesystem_leaves(path: str) -> list[str]:
    """Recursively find all files from path"""
    return reduce(lambda t, s: t + [s] if not s[1] else t, os.walk(path), [])


def compose(*funs):
    """Function composition e.g. g(f(x))"""

    def wrapper(args):
        return reduce(lambda r, f: f(r), funs, args)

    return wrapper


def pipe(*args):
    """Pipe data through functions"""
    return reduce(lambda r, f: f(r), rest(args), first(args))


def thread(*args):
    """Pipe data in a list style formatting"""
    return reduce(lambda r, f: f[0](r, *rest(f)), rest(args), first(args))


def printr(x, fun=lambda x: print(x)):
    """Debugging statement for piping"""
    fun(x)
    return x


def trace(f):
    def wrapper(*args, **kwargs):
        print(f"{f.__name__}: positional {args}, named {kwargs}")
        out = f(*args, **kwargs)
        print(f"=> {out}")
        return out
    return wrapper
    

def print_nested_sizes(x, depth=0):
    this_len = len(x)
    print(f"|{'-'*depth}: {this_len}")
    if isinstance(x, (list, tuple)):
        print_nested_sizes(x[0], depth=depth+1)


#################################################################################
#                           Record transformations                              #
#################################################################################


def label_record(labels, record):
    """Label a record"""
    return tuple(zip(labels, record))


def label_records(labels, records):
    """Label many records"""
    return tmap(lambda record: label_record(labels, record), records)


def pluck_item(name: Union[str, list[str]], iterable: Records) -> Any:
    """Get value corresponding to the key `name` of a record.

    :param name: The name of the `key` to retrieve the value from.
    :param iterable: The record or class to retrieve from.
    :type name: str
    :type iterable: Record or dataclass.
    :returns: The value corresponding with the found key. If the key
        is not found then None.

    >>> record = (('a', 2), ('b', 1))
    >>> pluck_item('a', record)
    2

    Pluck item will return the last value if there are duplicate
    keys. This is to allow the tracking of historical changes. Take
    for example:

    >>> new_record = add(record, "b", 3)
    >>> new_record
    (('a', 2), ('b', 1), ('b', 3))

    We see that `new_record` contains two duplicate keys, the latter
    of which is the new one we just added. When we use `pluck_item`,
    the most recent key will be used. Therefore, one can view the
    `add` function as also an update method.

    >>> pluck_item('b', new_record)
    3

    The `name` to pluck from the `iterable` can also be an iterable
    within itself. That means you can provide nested keys to pluck.

    >>> new_record = (('a', ('b', 3),)), ('b', 1))
    >>> pluck_item(['a', 'b'], new_record)
    3
    """
    if type(iterable) not in [tuple, list]:
        iterable = dataclass_to_record(iterable)
    if isinstance(name, list):
        if len(name) == 1:
            return pluck_item(first(name), iterable)
        return pluck_item(rest(name), pluck_item(first(name), iterable))
    return reduce(
        lambda val, item: item[1] if item[0] == name else val, iterable, None)


def pluck_items(names: list[str], iterable: Records) -> tuple[Any]:
    """Pluck many items for a single record"""
    return tmap(lambda name: pluck_item(name, iterable), names)


def pluck_list(name: str, iterable: Records) -> tuple[Any]:
    """Pluck an item from many records"""
    return tmap(lambda row: pluck_item(name, row), iterable)


def pluck_first(name: str, iterable):
    """Pluck a name from many records and return the first"""
    return first(pluck_list(name, iterable))


def keys(record: Records) -> tuple[str]:
    """Return all the keys available in the record

    :param record: The record to return keys from.
    :returns: The keys.

    >>> from src.dfp import keys
    >>> record = (('a', 1), ('b', 2))
    >>> keys(record)
    ('a', 'b')

    Keys will only return the unique set of keys. Therefore,
    if you've added multiple keys of the same name, these
    'duplicate' keys will only appear once.

    >>> record = (('a', 1), ('a', 2))
    >>> keys(record)
    ('a',)
    """
    return orderedset(tmap(lambda row: first(row), record))


def add(record, key, value):
    """Add a field to a record in a non-destructive way.

    :param record: The record to add a field to.
    :param key: The name of the new field.
    :param value: The value of the new field.
    :type record: tuple[tuple]
    :type key: str
    :type value: Any
    :returns: a new record with the field added.

    >>> record = (('a', 1), ('b', 2))
    >>> add(record, 'c', 3)
    (('a', 1), ('b', 2), ('c', 3))
    >>> record
    (('a', 1), ('b', 2))

    If the key already exists, another tuple is added, therefore
    preserving the history of changes:

    >>> add(record, 'a', 2)
    (('a', 1), ('b', 2), ('a', 2))

    When used in conjunction with pluck_item/pluck_list, one needn't
    worry about the duplicated keys however, as the 'latest' value
    will be returned.

    >>> pluck_item('a', add(record, 'a', 2))
    2
    """
    return (*record, (key, value))


def update(record, key, fn):
    return add(record, key, fn(pluck_item(key, record)))


def remove(record, key):
    """Remove a field from a record in a non-destructive way.

    :param record: The record to remove a field from.
    :param key: The name of the field to remove.
    :type record: tuple[tuple]
    :type key: str
    :returns: A new record with the field removed.

    >>> record = (('a', 1), ('b', 2), ('c', 3))
    >>> remove(record, 'c')
    (('a', 1), ('b', 2))
    """
    return tfilter(lambda x: first(x) != key, record)


def inverse(fun):
    """Inverse a boolean function

    Parameters
    ----------
    fun : Callable
        The function to inverse the result of

    Returns
    -------
    Callable
        A new function that provides the inverse

    Examples
    --------
    >>> inverse(lambda: False)()
    True
    >>> inverse(lambda x: x>0)(1)
    False

    """

    def wrapper(*args, **kwargs):
        return not fun(*args, **kwargs)
    return wrapper


def has_props(record, where) -> bool:
    """Check if record has a value with key"""
    def _check_iterable(k, v):
        return pluck_item(k, record) in v

    def _check_singleton(k, v):
        return pluck_item(k, record) == v

    def _check_function(k, v):
        return v(pluck_item(k, record))

    def _check(k, v):
        if isinstance(v, Iterable):
            return _check_iterable(k, v)
        elif callable(v):
            return _check_function(k, v)
        else:
            return _check_singleton(k, v)

    return all(tmap(lambda cond: _check(*cond), where.items()))


###################################################################################
#                               Type conversions                                  #
###################################################################################


def dataclass_to_record(dc):
    """Convert a dataclass to a record"""
    return tuple(dc._asdict().items())


def record_to_dataclass(dc_type, record):
    """Convert a record to a dataclass"""
    return dc_type(**dict(record))


def records_to_dataframe(records) -> pd.DataFrame:
    """Convert records to dataframe. (WARNING) assumes all records has the same keys"""
    names = orderedset(treduce(lambda t, s: t+list(keys(s)), records, []))
    data: dict[str, Any]  = {name: [] for name in names}
    for_each(
        lambda row: for_each(
            lambda col: data[first(col)].append(second(col)),
            row),
        records)
    return pd.DataFrame(data)


def dataframe_to_records(df) -> tuple:
    """Convert a dataframe to records"""
    names = list(df.columns)
    return tmap(
        lambda row: tmap(lambda col: (col, getattr(row[1], col)), names),
        df.iterrows())


def lz_tmap(fun):
    return lambda iterable: tmap(fun, iterable)
