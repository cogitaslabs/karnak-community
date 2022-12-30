import inspect


def inspect_caller_name(levels: int = 1) -> str:
    return inspect.stack()[levels][3]

