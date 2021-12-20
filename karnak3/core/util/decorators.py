import functools
import threading
from typing import Any, Callable


#
# Threading
#

def synchronized(wrapped: Callable[..., Any]) -> Any:
    """The missing @synchronized decorator
    https://git.io/vydTA"""
    _lock = threading.RLock()

    @functools.wraps(wrapped)
    def _wrapper(*args, **kwargs):
        nonlocal _lock
        with _lock:
            return wrapped(*args, **kwargs)

    return _wrapper
