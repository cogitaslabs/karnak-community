import datetime
from datetime import datetime as kdatetime
from datetime import date as kdate
from datetime import timedelta
import re
from typing import Optional, Union, List

import dateutil.parser
import tzlocal
import pytz

from karnak3.core.util import is_list_like


#
# type aliases
#

kdatelike = Union[kdate, kdatetime]


#
# time delta string parser
#

class TimeDeltaType(object):
    """
    Interprets a string as a timedelta for argument parsing.
    With no default unit:
    >>> tdtype = TimeDeltaType()
    >>> tdtype('5s')
    datetime.timedelta(0, 5)
    >>> tdtype('5.5s')
    datetime.timedelta(0, 5, 500000)
    >>> tdtype('5:06:07:08s')
    datetime.timedelta(5, 22028)
    >>> tdtype('5d06h07m08s')
    datetime.timedelta(5, 22028)
    >>> tdtype('5d')
    datetime.timedelta(5)
    With a default unit of minutes:
    >>> tdmins = TimeDeltaType('m')
    >>> tdmins('5s')
    datetime.timedelta(0, 5)
    >>> tdmins('5')
    datetime.timedelta(0, 300)
    >>> tdmins('6:05')
    datetime.timedelta(0, 21900)
    And some error cases:
    >>> tdtype('5')
    Traceback (most recent call last):
        ...
    ValueError: Cannot infer units for '5'
    >>> tdtype('5:5d')
    Traceback (most recent call last):
        ...
    ValueError: Colon not handled for unit 'd'
    >>> tdtype('5:5ms')
    Traceback (most recent call last):
        ...
    ValueError: Colon not handled for unit 'ms'
    >>> tdtype('5q')
    Traceback (most recent call last):
        ...
    ValueError: Unknown unit: 'q'
    """

    units = {
        'd': datetime.timedelta(days=1),
        'h': datetime.timedelta(seconds=60 * 60),
        'm': datetime.timedelta(seconds=60),
        's': datetime.timedelta(seconds=1),
        'ms': datetime.timedelta(microseconds=1000),
    }
    colon_mult_ind = ['h', 'm', 's']
    colon_mults = [24, 60, 60]
    unit_re = re.compile(r'[^\d:.,-]+', re.UNICODE)

    def __init__(self, default_unit=None):
        self.default_unit = default_unit

    def __call__(self, val):
        res = datetime.timedelta()
        for num, unit in self._parse(val):
            unit = unit.lower()

            if ':' in num:
                try:
                    colon_mults = self.colon_mults[:self.colon_mult_ind.index(unit) + 1]
                except ValueError:
                    raise ValueError('Colon not handled for unit %r' % unit)
            else:
                colon_mults = []

            try:
                unit = self.units[unit]
            except KeyError:
                raise ValueError('Unknown unit: %r' % unit)

            mult = 1
            for part in reversed(num.split(':')):
                res += self._mult_td(unit, (float(part) if '.' in part else int(part)) * mult)
                if colon_mults:
                    mult *= colon_mults.pop()
        return res

    def _parse(self, val):
        pairs = []
        start = 0
        for match in self.unit_re.finditer(val):
            num = val[start:match.start()]
            unit = match.group()
            pairs.append((num, unit))
            start = match.end()
        num = val[start:]
        if num:
            if pairs or self.default_unit is None:
                raise ValueError('Cannot infer units for %r' % num)
            else:
                pairs.append((num, self.default_unit))
        return pairs

    @staticmethod
    def _mult_td(td, mult):
        # Necessary because timedelta * float is not supported:
        return datetime.timedelta(days=td.days * mult, seconds=td.seconds * mult,
                                  microseconds=td.microseconds * mult)


default_tz = pytz.utc


def set_default_tz(tz=None, local=True):
    global default_tz
    if local:
        default_tz = tzlocal.get_localzone()
    elif tz is not None:
        default_tz = tz
    else:
        default_tz = pytz.utc


def kts(year, month, day, hour=0, minute=0, second=0, microsecond=0, tz=default_tz) \
        -> datetime.datetime:
    # timezone aware timestamp
    ts = datetime.datetime(year, month, day, hour, minute, second, microsecond, tzinfo=tz)
    return ts


def is_tz_aware(ts):
    """Test if timestamp is timezone aware"""
    return ts.tzinfo is not None and ts.tzinfo.utcoffset(ts) is not None


def to_tz_aware(ts: Union[kdatelike, List[kdatelike]], tz=default_tz) \
        -> Union[kdatetime, List[kdatetime], None]:
    if tz is None:
        if default_tz is None:
            set_default_tz()
        tz = default_tz

    if is_list_like(ts):
        return [to_tz_aware(t) for t in ts]
    if ts is None:
        return None

    if not isinstance(ts, kdatetime):
        _ts = dt_to_ts(ts, tz=tz)
    else:
        _ts = ts

    if is_tz_aware(_ts):
        return _ts
    else:
        return _ts.replace(tzinfo=tz)


def ts_to_end_of_day(ts: datetime.datetime):
    t = datetime.time.max
    return kdatetime(ts.year, ts.month, ts.day,
                     t.hour.t.minute, t.second, t.microsecond,
                     tzinfo=ts.tzinfo)


def ts_to_dt_end_of_day(ts: kdatelike) -> kdate:
    """convert to same date if 0:0:0.0, else next day"""
    _ts = ts
    if not isinstance(ts, kdatetime):
        _ts = dt_to_ts(ts)
    ts_prev = _ts - timedelta(milliseconds=1)
    dt = ts_to_dt(ts_prev) + timedelta(days=1)
    return dt

#
# def ts_to_start_of_day(dt: datetime.datetime):
#     t = datetime.time.min
#     return kts(dt.year, dt.month, dt.day,
#                t.hour, t.minute, t.second, t.microsecond,
#                tz=dt.tzinfo)


def dt_next_month(dt: datetime.date) -> datetime.date:
    month_start = datetime.date(dt.year, dt.month, 1)
    next_month_tmp = month_start + datetime.timedelta(days=32)
    next_month = datetime.date(next_month_tmp.year, next_month_tmp.month, 1)
    return next_month


def dt_month_last_day(dt: datetime.date) -> datetime.date:
    return dt_next_month(dt) - datetime.timedelta(days=1)


def dt_to_ts(dt: datetime.date, tz=None) -> datetime.datetime:
    return kts(dt.year, dt.month, dt.day, tz=tz)


def ts_to_dt(ts: datetime.date) -> datetime.date:
    return datetime.date(ts.year, ts.month, ts.day)


def now(tz=None):
    return datetime.datetime.now(tz=tz)


def today(tz=None):
    return ts_to_dt(now(tz))


def yesterday(tz=None):
    return today(tz) - datetime.timedelta(days=1)


def tomorrow(tz=None):
    return today(tz) + datetime.timedelta(days=1)


#
# filename parse
#


def parse_filename_date(filename) -> Optional[datetime.date]:
    match = re.search(r'\d{8}', filename)
    if match:
        dt_str = match.group(0)
        ts = datetime.datetime.strptime(dt_str, '%Y%m%d')
        dt = ts_to_dt(ts)
        return dt
    else:
        return None


def parse_filename_datetime(filename, tz=default_tz) -> Optional[datetime.datetime]:
    match_expressions = [r'\d{8}[-:\s]?\d{6}', r'\d{8}']
    for exp in match_expressions:
        match = re.search(exp, filename)
        if match:
            ts_str = match.group(0)
            ts = dateutil.parser.isoparse(ts_str)
            ts_aware = to_tz_aware(ts, tz)
            return ts_aware
    return None


#
# weekdays
#

valid_weekday_str = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']


def weekday_str_to_int(weekday_str: Optional[str]) -> Optional[int]:
    if weekday_str is None or not weekday_str.lower() in valid_weekday_str:
        return None
    return valid_weekday_str.index(weekday_str.lower())
