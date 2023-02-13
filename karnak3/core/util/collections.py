import collections.abc
from typing import Iterable, List, Optional

import pandas as pd


# TODO review
def none_param2list(param):
    if param is None:
        return []
    else:
        return param2list(param)


# TODO review
def param2list(param):
    if type(param) in [list, set, tuple]:
        return list(param)
    else:
        return [param]


# TODO review
def list2str(l):
    return ' '.join([str(elem) for elem in l])


#
# *arg
#

def coalesce(*elems):
    if elems is None:
        return None
    for e in elems:
        if e is not None:
            return e
    return elems[-1]


def first_valid(*elements):
    if elements is None:
        return None
    for e in elements:
        if e is not None:
            return e
    return None


def safe_join(*elements: str, sep='') -> str:
    """Generic string concatenator. Ignores None elements"""
    non_none = [e for e in elements if e is not None]
    return sep.join(non_none)


def is_list_like(obj) -> bool:
    return is_sequence(obj)


def is_sequence(obj) -> bool:
    return obj is not None and isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str)


def is_map(obj) -> bool:
    return obj is not None and isinstance(obj, collections.abc.Mapping)


def remove_none(it: Iterable) -> List:
    ret = [x for x in it if x is not None]
    return ret


def remove_na(it: Iterable) -> List:
    ret = [x for x in it if x is not None and not pd.isna(x)]
    return ret


def safe_get(d: Optional[collections.abc.Mapping], *keys):
    if not is_map(d):
        return None
    ret = d
    for k in keys:
        if ret is None:
            return None
        ret = ret.get(k)
    return ret
