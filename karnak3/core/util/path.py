import errno
import os
from typing import Optional


def path_concat(path: Optional[str], *elements: Optional[str]) -> str:
    """safely joins path elements, discarding redundant slashes ('/')"""
    if not path and not elements:
        return ''
    base_path = [path.strip().rstrip('/')] if path is not None else []
    safe_elements = base_path + [e.strip().rstrip('/').lstrip('/')for e in elements if
                                 e is not None and e.strip().rstrip('/').lstrip('/')]

    return '/'.join(safe_elements)


def create_folder(filepath: str) -> None:
    if not os.path.exists(os.path.dirname(filepath)):
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
