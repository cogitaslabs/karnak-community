from typing import Union, Optional, Tuple, List

from datetime import timedelta

import karnak3.core.util as ku
from karnak3.core.util import kdatetime, kdate, kdatelike


valid_slicing_frequency = ['hour', 'day', 'month', 'year',
                           '30min', '20min', '15min', '10min', '5min', 'minute']

valid_latest_frequency = ['mtd', 'ytd', 'full', 'latest']

valid_overlapping_frequency = ['adhoc', 'hourly', 'daily', 'none'] + valid_latest_frequency

valid_frequency = valid_slicing_frequency + valid_overlapping_frequency

valid_auto_window_basic = ['full', 'last-year', 'ytd', 'last-month', 'mtd', 'yesterday', 'today',
                           'latest', 'filename']

# valid_auto_window_bookmark = ['bookmark', 'leftover']

valid_auto_window = valid_auto_window_basic # + valid_auto_window_bookmark


#
# helper
#


def decode_auto_window(auto_window: str,
                       tz=ku.default_tz,
                       days: Optional[int] = None,
                       filename: Optional[str] = None) \
        -> Tuple[Optional[kdatetime], Optional[kdatetime]]:

    # this does not handle bookmarks
    assert auto_window in valid_auto_window_basic

    # date windows are inclusive in start, exclusive (not included) in end
    # that means: if window_end is today, than the last day included in window is yesterday

    # helpers
    today = ku.dt_to_ts(ku.today(tz), tz=tz)
    tomorrow = ku.dt_to_ts(ku.today(tz), tz=tz) + timedelta(days=1)
    yesterday = ku.dt_to_ts(ku.yesterday(tz), tz=tz)
    current_year = today.year
    current_month = today.month
    previous_month = current_month - 1
    previous_month_year = current_year
    if previous_month == 0:
        previous_month = 12
        previous_month_year = current_year - 1

    if auto_window == 'full':
        start_date = None
        end_date = None
    elif auto_window == 'last-year':
        start_date = ku.kts(current_year - 1, 1, 1, tz=tz)
        end_date = ku.kts(current_year, 1, 1, tz=tz)
    elif auto_window == 'ytd':
        start_date = ku.kts(current_year, 1, 1, tz=tz)
        end_date = today
    elif auto_window == 'last-month':
        start_date = ku.kts(previous_month_year, previous_month, 1, tz=tz)
        end_date = ku.kts(current_year, current_month, 1, tz=tz)
    elif auto_window == 'mtd':
        start_date = ku.kts(current_year, current_month, 1, tz=tz)
        end_date = today
    elif auto_window in ['yesterday', 'today']:
        _days = 1 if days is None else days
        if auto_window == 'today':
            end_date = tomorrow
        elif auto_window == 'yesterday':
            end_date = today
        else:
            assert False
        start_date = end_date - timedelta(days=_days)
    elif auto_window == 'filename' and filename is not None:
        filename_ts = ku.parse_filename_datetime(filename, tz)
        return filename_ts, filename_ts
    else:
        assert False

    return start_date, end_date


def slice_time_window(window_start: Optional[kdatelike] = None,
                      window_end:  Optional[kdatelike] = None,
                      weekdays: Optional[List[str]] = None,
                      last_weekday: Optional[str] = None,
                      frequency: Optional[str] = None,
                      tz=ku.default_tz,
                      time_type: str = 'datetime') \
        -> List[Tuple[kdatelike, kdatelike]]:

    def filter_weekdays(lst: List[Tuple[kdatetime, kdatetime]]) \
            -> List[Tuple[kdatetime, kdatetime]]:
        if (weekdays is None and last_weekday is None) or frequency != 'day':
            return lst
        weekdays_included_str = weekdays if weekdays is not None else [last_weekday]
        weekdays_included = [ku.weekday_str_to_int(d) for d in weekdays_included_str]
        filtered_lst = [t for t in lst if t[0].weekday() in weekdays_included]

        if last_weekday is not None:
            dt_start_list = [ku.ts_to_dt(t1) for (t1, t2) in filtered_lst]
            max_date = max(dt_start_list)
            double_filtered_lst = [t for t in filtered_lst if ku.ts_to_dt(t[0]) == max_date]
            return double_filtered_lst
        else:
            return filtered_lst

    def format_result(lst: List[Tuple[kdatetime, kdatetime]]) \
            -> List[Tuple[kdatelike, kdatelike]]:
        ret = filter_weekdays(lst)
        if time_type == 'date':
            ret = [(ku.ts_to_dt(xs), ku.ts_to_dt_end_of_day(xe)) for (xs, xe) in ret]
        return ret

    w_start = window_start
    w_end = window_end
    if w_start is None and w_end is None:
        # empty window. in this case, both limits should be None
        return []

    # overlapping frequencies do not get sliced
    if frequency in valid_overlapping_frequency or frequency is None:
        # or auto_window == 'leftover':
        return format_result([(w_start, w_end)])

    # window start and window end must be both defined or both undefined
    assert w_start is not None and w_end is not None

    w_end_adj = w_end - timedelta(milliseconds=1)

    if w_start >= w_end:
        # start > end
        return []

    elif frequency == 'year':
        # always whole years, from first to last day
        first = w_start.year
        last = w_end_adj.year
        slices = []
        for y in range(first, last+1):
            year_start = ku.kts(y, 1, 1, tz=tz)
            year_end = ku.kts(y+1, 1, 1, tz=tz)
            slices.append((year_start, year_end))
        return format_result(slices)

    elif frequency == 'month':
        # always whole months, from first to last day
        first = kdate(w_start.year, w_start.month, 1)
        last = kdate(w_end_adj.year, w_end_adj.month, 1)
        slices = []
        m = first
        while m <= last:
            month_start = ku.kts(m.year, m.month, 1, tz=tz)
            month_end = ku.dt_to_ts(ku.dt_next_month(month_start), tz)
            slices.append((month_start, month_end))
            m = ku.dt_next_month(m)
        return format_result(slices)

    elif frequency == 'day':
        first_date = w_start.date()
        last = kdate(w_end_adj.year, w_end_adj.month, w_end_adj.day)
        slices = []
        dt = first_date
        while dt <= last:
            next_dt = dt + timedelta(days=1)
            ts = ku.dt_to_ts(dt, tz)
            next_ts = ku.dt_to_ts(next_dt, tz)
            slices.append((ts, next_ts))
            dt = next_dt
        return format_result(slices)

    elif frequency == 'hour':
        first = ku.kts(w_start.year, w_start.month, w_start.day, w_start.hour, tz=tz)
        last = ku.kts(w_end_adj.year, w_end_adj.month, w_end_adj.day, w_end_adj.hour, tz=tz)
        slices = []
        ts = first
        while ts <= last:
            next_ts = ts + timedelta(hours=1)
            slices.append((ts, next_ts))
            ts = next_ts
        return format_result(slices)

    elif frequency == 'minute':
        first = ku.kts(w_start.year, w_start.month, w_start.day, w_start.hour,
                       w_start.minute, tz=tz)
        last = ku.kts(w_end_adj.year, w_end_adj.month, w_end_adj.day, w_end_adj.hour,
                      w_end_adj.minute, tz=tz)
        slices = []
        ts = first
        while ts <= last:
            next_ts = ts + timedelta(minutes=1)
            slices.append((ts, next_ts))
            ts = next_ts
        return format_result(slices)

    elif frequency in ['5min', '15min', '20min', '30min']:
        x_min = int(frequency.replace('min', ''))
        first = ku.kts(w_start.year, w_start.month, w_start.day, w_start.hour,
                       w_start.minute, tz=tz)
        last = ku.kts(w_end_adj.year, w_end_adj.month, w_end_adj.day, w_end_adj.hour,
                      w_end_adj.minute, tz=tz)
        slices = []
        ts = first
        while ts <= last:
            next_ts = ts + timedelta(minutes=x_min)
            slices.append((ts, next_ts))
            ts = next_ts
        return format_result(slices)

    else:
        assert False


def decode_time_window_limits(window_start: Optional[kdatelike] = None,
                              window_end: Optional[kdatelike] = None,
                              window_date: Optional[kdate] = None,
                              auto_window: Optional[str] = None,
                              days: Optional[int] = None,
                              year: Optional[int] = None,
                              # weekdays: Optional[List[str]] = None,
                              # last_weekday: Optional[str] = None,
                              filename: Optional[str] = None,
                              frequency: Optional[str] = None,
                              tz=ku.default_tz,
                              time_type: str = 'datetime') \
        -> Union[Tuple[kdatetime, kdatetime],
                 Tuple[kdate, kdate]]:

    # window_end is always exclusive (i.e., not included)

    assert time_type in ['datetime', 'date']
    assert frequency in valid_frequency or frequency is None
    assert auto_window in valid_auto_window or auto_window is None

    # helper
    today = ku.to_tz_aware(ku.today(tz), tz)
    w_start = ku.to_tz_aware(window_start, tz)
    w_end = ku.to_tz_aware(window_end, tz)

    if window_start and window_end:
        pass  # nothing to do
    # elif auto_window in ['bookmark', 'leftover']:
    #     w_start, w_end = decode_auto_window_bookmarks(w_start, frequency, auto_window, cooling, tz,
    #                                                   bookmark_group=bookmark_group,
    #                                                   bookmark_name=bookmark_name)
    elif window_date:
        w_start = ku.dt_to_ts(window_date, tz=tz)
        w_end = w_start + timedelta(days=1)
    elif auto_window:
        w_start, w_end = decode_auto_window(auto_window, tz=tz, days=days, filename=filename)
    elif window_start and days:
        w_end = w_start + timedelta(days=days)
    elif window_end and days:
        w_start = w_end - timedelta(days=days)
    elif days:
        w_end = today
        w_start = w_end - timedelta(days=days)
    elif year:
        w_start = ku.kts(year, 1, 1, tz=tz)
        w_end = ku.kts(year+1, 1, 1, tz=tz)
        if today < w_end:
            w_end = today
    else:
        raise ku.KarnakException('not enough parameters to determine time window limits.')

    _w_start = ku.ts_to_dt(w_start) if time_type == 'date' else w_start
    _w_end = ku.ts_to_dt_end_of_day(w_end) if time_type == 'date' else w_end
    return _w_start, _w_end


def decode_time_window_slices(window_start: Optional[kdatelike] = None,
                              window_end: Optional[kdatelike] = None,
                              window_date: Optional[kdate] = None,
                              auto_window: Optional[str] = None,
                              days: Optional[int] = None,
                              year: Optional[int] = None,
                              weekdays: Optional[List[str]] = None,
                              last_weekday: Optional[str] = None,
                              filename: Optional[str] = None,
                              frequency: Optional[str] = None,
                              tz=ku.default_tz,
                              time_type: str = 'datetime') \
        -> List[Tuple[kdatelike, kdatelike]]:
    ok = validate_time_window(window_start=window_start,
                              window_end=window_end,
                              window_date=window_date,
                              auto_window=auto_window,
                              days=days,
                              year=year,
                              weekdays=weekdays,
                              last_weekday=last_weekday,
                              filename=filename,
                              frequency=frequency)
    assert ok
    w_start, w_end = decode_time_window_limits(window_start=window_start,
                                               window_end=window_end,
                                               window_date=window_date,
                                               auto_window=auto_window,
                                               days=days,
                                               year=year,
                                               filename=filename,
                                               frequency=frequency,
                                               tz=tz,
                                               time_type='datetime')  # convert only in the end

    slices = slice_time_window(window_start=w_start,
                               window_end=w_end,
                               weekdays=weekdays,
                               last_weekday=last_weekday,
                               frequency=frequency,
                               tz=tz,
                               time_type=time_type)
    return slices


def time_window_slices_start(slices: List[Tuple[kdatelike, kdatelike]]) -> List[kdatelike]:
    return [ts for (ts, te) in slices]


def time_window_slices_end(slices: List[Tuple[kdatelike, kdatelike]]) -> List[kdatelike]:
    return [te for (ts, te) in slices]


def validate_time_window(window_start: Optional[kdatelike] = None,
                         window_end: Optional[kdatelike] = None,
                         window_date: Optional[kdate] = None,
                         auto_window: Optional[str] = None,
                         days: Optional[int] = None,
                         year: Optional[int] = None,
                         weekdays: Optional[List[str]] = None,
                         last_weekday: Optional[str] = None,
                         filename: Optional[str] = None,
                         frequency: Optional[str] = None,
                         time_type='datetime'
                         ) -> Tuple[bool, str]:
    # is window_end is not given, assumes if needed current timestamp or current date
    #   according to time_type

    assert time_type in ['date', 'datetime', None]
    assert frequency is None or frequency in valid_frequency
    assert auto_window is None or auto_window in valid_auto_window

    if window_start is not None and window_end is not None:
        if auto_window or days or year or window_date:
            return False, "'days', 'year', 'dt' cannot be used with " \
                          "'window_start' and 'window_end'"
        if window_end <= window_start:
            return False, "'window_start' must be before 'window_end'"

    if days is not None:
        if not any([window_start, window_end, window_date, auto_window]):
            return False, "'days' requires 'window_start', 'window_end', 'dt' ou 'auto_window'"
        if auto_window and auto_window not in ['yesterday', 'today']:
            return False, "'days' when used with 'auto_window' require 'today' or 'yesterday'"

    if window_end is not None:
        if days is None and window_start is None:
            return False, "'window_end' requires 'window_start' or 'days'"

    if year:
        if (window_start is None or window_end is not None or auto_window or weekdays
                or last_weekday):
            return False, "'year' cannot be used with windows"

    if window_date:
        if window_start or window_end or days or auto_window or weekdays or last_weekday:
            return False, "'dt' cannot be used with windows, 'days' or weekdays"

    if weekdays:
        if last_weekday:
            return False, "'weekdays' cannot be used with 'last_weekday'"
        if frequency in valid_slicing_frequency and frequency != ['day']:
            return False, f"'weekdays' cannot be used with frequency {frequency}"
        assert ku.is_list_like(weekdays)
        for x in weekdays:
            if ku.weekday_str_to_int(x) is None:
                return False, f"invalid weekday: '{x}'," \
                              f" Should be one of: {', '.join(ku.valid_weekday_str)}"

    if last_weekday:
        if frequency in valid_slicing_frequency and frequency != ['day']:
            return False, f"'last_weekday' cannot be used with frequency {frequency}"
        if ku.weekday_str_to_int(last_weekday) is None:
            return False, f"invalid weekday: '{last_weekday}'," \
                          f" Should be one of: {', '.join(ku.valid_weekday_str)}"

    if filename is not None:
        if auto_window != 'filename':
            return False, f"'filename' auto-window 'filename' must be used together"

    if auto_window == 'filename':
        if ku.str_empty(filename):
            return False, f"'filename' auto-window 'filename' must be used together"
        if frequency in valid_slicing_frequency:
            return False, f"'filename' auto-windows cannot be sliced"

    if (auto_window in ['mtd', 'last_month', 'yesterday', 'today']
                and frequency in [None, 'year', 'yearly']) \
            or (auto_window in ['yesterday', 'today'] and frequency in ['month', 'monthly']):
        # TODO there are more incompatible combinations
        return False, f"incompatible auto_window {auto_window} with frequency '{frequency}'"

    if time_type == 'date' and (frequency not in ['day', 'month', 'year']
                                and frequency in valid_slicing_frequency) \
            or frequency == 'monthly':
        return False, f"invalid frequency for date"

    if not any([window_start, window_end, auto_window, window_date, year]):
        return False, "no window criteria defined"

    return True, ''
