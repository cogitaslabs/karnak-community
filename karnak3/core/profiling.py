import datetime
import time
from typing import Dict, List, Union
import psutil
import sys
import gc
import pandas as pd
import resource  # noqa
import os

import karnak3.core.log as klog


def get_size(obj, seen=None) -> int:
    """Recursively finds size of objects
    https://gist.githubusercontent.com/bosswissam/a369b7a31d9dcab46b4a034be7d263b2/raw/f99d210019c1fac6bb46d2da81dcdf5ef9932172/pysize.py"""

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def pretty_mem(size_bytes: int, round_decimals: int = 3) -> str:
    size = size_bytes
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if abs(size) < 1024.0 or unit == 'PiB':
            break
        size = size / 1024.0

    if unit == 'B':
        ret = f'{size_bytes} B'
    else:
        ret = f'{size:.{round_decimals}f} {unit}'
    return ret


def df_memory_usage(df: Union[pd.DataFrame, pd.Series]) -> int:
    """dataframe memory usage in bytes"""
    if isinstance(df, pd.DataFrame):
        return df.memory_usage(index=True, deep=True).sum()
    elif isinstance(df, pd.Series):
        return df.memory_usage(index=True, deep=True)
    else:
        klog.warn('df_memory_usage: object is not a dataframe or series')
        return 0


def df_memory_usage_str(df: Union[pd.DataFrame, pd.Series], round_decimals: int = 2) -> str:
    return pretty_mem(df_memory_usage(df), round_decimals=round_decimals)


def memory_usage(run_gc: bool = True) -> Dict[str, int]:
    if run_gc:
        gc.collect()
    total_usage = resource.getrusage(resource.RUSAGE_SELF)
    mem_usage = psutil.virtual_memory()
    process_memory = psutil.Process(os.getpid()).memory_info()
    avail = mem_usage.available
    used = mem_usage.used
    if avail == used:
        avail = mem_usage.free
    usage = {
        'used': mem_usage.used,
        'avail': avail,
        'total': mem_usage.total,
        'p rss': process_memory.rss,  # process resident
        'p virt': process_memory.vms,  # process virtual
        'p maxrss': total_usage.ru_maxrss * 1024,  # process max resident
    }
    return usage


def delta_memory_usage(usage_before: Dict[str, int],
                       usage_after: Dict[str, int],
                       keys: List[str] = None) -> Dict[str, int]:
    _keys = keys if keys is not None else ['used', 'p rss', 'p virt', 'p maxrss']
    delta = {f'delta {k}': usage_after[k] - usage_before[k] for k in _keys}
    return delta


def _pretty_usage_str(usage: Dict[str, int], round_decimals: int = 2) -> str:
    usage_list_str = [f'{k}: {pretty_mem(usage[k], round_decimals)}' for k in usage]
    return ', '.join(usage_list_str)


def memory_usage_str(run_gc: bool = True, round_decimals: int = 2):
    usage = memory_usage(run_gc=run_gc)
    return _pretty_usage_str(usage, round_decimals=round_decimals)


class KTimer:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time

    def log_timer(self, message=None, end_time=None):
        if message is None:
            message = 'elapsed'
        klog.debug(f'{message} {self.get_elapsed_str(end_time)}')

    def log_delta_elapsed(self, message=None, end_time=None):
        if message is None:
            message = 'elapsed'
        klog.debug(f'{message} {self.get_delta_elapsed_str(end_time)}')

    def get_elapsed(self, end_time=None):
        if end_time is None:
            end_time = time.time()
        return datetime.timedelta(seconds=end_time - self.start_time)

    def get_elapsed_str(self, end_time=None):
        elapsed = self.get_elapsed(end_time)
        return str(elapsed)

    def get_delta_elapsed(self, end_time=None):
        if end_time is None:
            end_time = time.time()
        ret = datetime.timedelta(seconds=end_time - self.last_time)
        self.last_time = end_time
        return ret

    def get_delta_elapsed_str(self, end_time=None):
        elapsed = self.get_delta_elapsed(end_time)
        return str(elapsed)


class KProfiler(KTimer):
    def __init__(self, run_gc: bool = False):
        self.first_mem_usage = memory_usage(run_gc=run_gc)
        self.last_mem_usage = self.first_mem_usage
        super().__init__()

    def mem_str(self, run_gc: bool = False) -> str:
        mem_usage = memory_usage(run_gc=run_gc)
        self.last_mem_usage = mem_usage
        _pretty_mem = _pretty_usage_str(mem_usage)
        return _pretty_mem

    def log_mem(self, message=None, level='DEBUG', run_gc: bool = False):
        _pretty_mem = self.mem_str(run_gc=run_gc)
        _message = message + ': ' + _pretty_mem if message else _pretty_mem
        klog.log(level, _message)

    # FIXME implement delta_only
    def _log_difference(self, usage_before: Dict[str, int], message: str = None,
                        end_time: datetime.datetime = None,
                        level='TRACE', cumulative: bool = True,
                        delta_only: bool = False,
                        basic_level='DEBUG',
                        run_gc: bool = False):
        _, should_log = klog.get_log_threshold(level)
        elapsed_str = self.get_elapsed_str(end_time)
        delta_elapsed_str = self.get_delta_elapsed(end_time)
        cumulative_str = ''
        if cumulative:
            cumulative_str = f'total: {elapsed_str} '
        _difference_str = f'delta: {delta_elapsed_str} {cumulative_str}'
        if should_log:
            mem_usage = memory_usage(run_gc=run_gc)
            delta_usage = delta_memory_usage(usage_before, mem_usage)
            all_usage = mem_usage.copy()
            all_usage.update(delta_usage)
            if delta_only:
                all_usage = {k: all_usage[k] for k in all_usage if k.startswith('delta')}
            _pretty_usage = _pretty_usage_str(all_usage)
            _difference_usage_str = _difference_str + ' ' + _pretty_usage
            _message = message + ': ' + _difference_usage_str if message else _difference_usage_str
            klog.log(level, _message)
            self.last_mem_usage = mem_usage
        else:
            _message = message + ': ' + _difference_str if message else _difference_str
            klog.log(basic_level, _message)

    def log_delta(self, message: str = None,
                  end_time: datetime.datetime = None,
                  level='TRACE',
                  delta_only: bool = False,
                  basic_level='DEBUG',
                  run_gc: bool = False):
        self._log_difference(self.last_mem_usage, message, end_time, level, delta_only=delta_only,
                             basic_level=basic_level, run_gc=run_gc)

    def log_cumulative(self, message: str = None,
                       end_time: datetime.datetime = None, level='TRACE',
                       delta_only: bool = False,
                       run_gc: bool = False):
        self._log_difference(self.first_mem_usage, message, end_time, level, cumulative=True,
                             delta_only=delta_only, run_gc=run_gc)

    def log_profile(self, message: str = None,
                    end_time: datetime.datetime = None,
                    run_gc: bool = False):
        self.log_cumulative(message=message, end_time=end_time, run_gc=run_gc)
