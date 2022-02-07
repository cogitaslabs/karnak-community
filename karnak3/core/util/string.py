from typing import Optional


def str_empty(s: Optional[str]) -> bool:
    return s is not None and s.strip() != ''


def str_ensure_prefix(s: str, prefix: str):
    if s is None:
        return None
    if s.startswith(prefix):
        return s
    else:
        return prefix + s


def str_ensure_suffix(s: str, suffix: str):
    if s is None:
        return None
    if s.endswith(suffix):
        return s
    else:
        return s + suffix


def str_ensure_no_suffix(s: str, suffix: str):
    if s is None:
        return None
    if s.endswith(suffix):
        return s[:-len(suffix)]
    else:
        return s


