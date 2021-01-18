import os
import errno
from typing import List


def path_concat(path: str, *elements: str) -> str:
    base_path = path.rstrip('/')
    safe_elements = [base_path] + [e.rstrip('/').lstrip('/') for e in elements if
                                   e is not None and e.rstrip('/').lstrip('/') != '']
    return '/'.join(safe_elements)


def create_folder(filepath: str) -> None:
    if not os.path.exists(os.path.dirname(filepath)):
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
