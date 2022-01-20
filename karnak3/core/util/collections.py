import collections.abc


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
    return isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str)
