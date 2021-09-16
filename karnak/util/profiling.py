import datetime
import time
import karnak.util.log as klog
import psutil
import sys


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


class KTimer:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time

    def log_timer(self, message=None, end_time=None):
        if message is None:
            message = 'elapsed'
        klog.debug('{} {}', message, self.get_elapsed_str(end_time))

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
    def __init__(self):
        mem_usage = psutil.virtual_memory()
        self.start_mem_available = mem_usage.available
        self.start_mem_used = mem_usage.used
        self.start_mem_free = mem_usage.free
        self.start_mem_total = mem_usage.total

        self.last_mem_available = mem_usage.available
        self.last_mem_used = mem_usage.used
        self.last_mem_free = mem_usage.free
        self.last_mem_total = mem_usage.total
        super().__init__()

    @staticmethod
    def pretty_mem(size_bytes, round_decimals=3) -> str:
        size = size_bytes
        for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
            if abs(size) < 1024.0 or unit == 'PiB':
                break
            size = size / 1024.0
        return f"{size:.{round_decimals}f} {unit}"

    @staticmethod
    def log_mem(message=None):
        pretty_mem = KProfiler.pretty_mem
        mem_usage = psutil.virtual_memory()
        msg = ''
        if message is not None:
            msg = message + ': '
        klog.debug(f'{msg}mem avail {pretty_mem(mem_usage.available)}, '
                   f'used {pretty_mem(mem_usage.available)}, '
                   f'free {pretty_mem(mem_usage.free)}, '
                   f'total {pretty_mem( mem_usage.total)}')

    def log_delta(self, message=None, end_time=None):
        msg = ''
        if message is not None:
            msg = message + ': '

        delta_elapsed_str = self.get_delta_elapsed_str(end_time)
        total_elapsed_str = self.get_elapsed_str(end_time)
        mem_usage = psutil.virtual_memory()

        delta_avail_str = self.pretty_mem(mem_usage.available - self.last_mem_available)
        delta_used_str = self.pretty_mem(mem_usage.used - self.last_mem_used)

        klog.debug(f'{msg}elapsed {delta_elapsed_str}s, '
                   f'total elapsed {total_elapsed_str}, '
                   f'mem: delta avail {delta_avail_str}, '
                   f'delta used {delta_used_str}, '
                   f'avail {self.pretty_mem(mem_usage.available)}, '
                   f'used {self.pretty_mem(mem_usage.used)}, '
                   f'free {self.pretty_mem(mem_usage.free)}, '
                   f'total {self.pretty_mem(mem_usage.total)}')

        self.last_mem_available = mem_usage.available
        self.last_mem_used = mem_usage.used
        self.last_mem_free = mem_usage.free
        self.last_mem_total = mem_usage.total

    def log_cumulative(self, message=None, end_time=None):
        msg = ''
        if message is not None:
            msg = message + ': '

        elapsed_str = self.get_elapsed_str(end_time)
        mem_usage = psutil.virtual_memory()
        delta_avail_str = self.pretty_mem(mem_usage.available - self.start_mem_available)
        delta_used_str = self.pretty_mem(mem_usage.used - self.start_mem_used)

        klog.debug(f'{msg}elapsed {elapsed_str}s, '
                   f'mem: delta avail {delta_avail_str}, '
                   f'delta used {delta_used_str}, '
                   f'avail {self.pretty_mem(mem_usage.available)}, '
                   f'used {self.pretty_mem(mem_usage.used)}, '
                   f'free {self.pretty_mem(mem_usage.free)}, '
                   f'total {self.pretty_mem(mem_usage.total)}')

    def log_profile(self, message=None, end_time=None):
        self.log_cumulative(message=message, end_time=end_time)
