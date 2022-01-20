from typing import Optional


def str_empty(s: Optional[str]) -> bool:
    return s is not None and s.strip() != ''


