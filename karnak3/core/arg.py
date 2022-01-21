from typing import Optional, Dict, List, Union, Tuple
import argparse
import datetime
import dateutil.parser
import pytz
import os
import copy

import karnak3.core.util as ku
import karnak3.core.log as kl
import karnak3.core.time_window as ktw
from karnak3.core.util.datetime import kdatelike


def best_arg(arg_name: str, args: Optional[Dict[str, str]] = None,
             env_var: str = None,
             default: str = None) -> Optional[str]:
    """looks for argument value in arguments, then environment"""
    arg_value = None
    if args is not None:
        arg_value = args.get(arg_name)
    if arg_value is None:
        _env_var = env_var if env_var is not None else f'KARNAK_{arg_name.upper()}'
        arg_value = os.environ.get(_env_var)
    if arg_value is None:
        arg_value = default
    return arg_value


def parse_date(date_str):
    try:
        return datetime.datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise argparse.ArgumentTypeError(f'date {date_str} not in format YYYY-MM-DD')


def parse_timestamp(s):
    try:
        return dateutil.parser.isoparse(s)
    except Exception:
        raise argparse.ArgumentTypeError(f'not a valid timestamp: {s}')


def parse_timezone(s):
    try:
        return pytz.timezone(s)
    except Exception:
        raise argparse.ArgumentTypeError(f'not a valid timezone: {s}')


def parse_dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")


def parse_file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid file")


def test_argument_added(parser: argparse.ArgumentParser, argument_name: str) -> bool:
    second_parser = copy.deepcopy(parser)
    try:
        second_parser.add_argument(argument_name)
        return False
    except Exception as e:
        return True


#
# core parameters
#

def add_core_params(parser: argparse.ArgumentParser):
    parser.add_argument('--verbosity', '-v', type=int, default=2,
                        help='log verbosity: 0 is less verbose, 5 is more verbose')
    parser.add_argument('--timezone', '-z', help='default timezone for semantic timestamp',
                        type=parse_timezone, default=None)


def parse_handle_core_params(parser: argparse.ArgumentParser, arguments):
    # KarnakConfig.module_name = module_name
    args = parser.parse_args(arguments)

    kl.set_level(args.verbosity)
    if args.timezone is not None:
        ku.set_default_tz(args.timezone)


#
# time windows
#

def add_window_params(parser: argparse.ArgumentParser, time_type: str = 'timestamp'):
    assert time_type in ['date', 'timestamp']

    if not test_argument_added(parser, '--timezone'):
        add_core_params(parser)

    time_parser = parse_timestamp if time_type == 'timestamp' else parse_date

    if time_type == 'date':
        parser.add_argument('--window-date', type=time_parser,
                            help='time window consisting of a single date a single date')
        parser.add_argument('--weekdays', nargs='+', type=str, choices=ku.valid_weekday_str,
                            help='select only chosen weekdays in time window')
        parser.add_argument('--last-weekday', type=str, choices=ku.valid_weekday_str,
                            help='select only last weekday as time window')

    parser.add_argument('--window-start', '--ws', type=time_parser,
                        help='start of time window, inclusive.')
    parser.add_argument('--window-end', '--we', type=time_parser,
                        help='end of time window, exclusive (date not included in range)')
    parser.add_argument('--auto-window', '-w', type=str, choices=ktw.valid_auto_window,
                        help='automatic window calculation')
    parser.add_argument('--days', type=int, default=1,
                        help='shortcut to set number of days to fetch. '
                             'Ignored if both window start or window end is present '
                             'or period is not daily.')
    parser.add_argument('--year', help='shortcut to download all entries for a chosen year.',
                        type=int)


def parse_window_slices(parsed_args,
                        window_start: Union[datetime.datetime, datetime.date, None] = None,
                        window_end: Union[datetime.datetime, datetime.date, None] = None,
                        window_date: Optional[datetime.date] = None,
                        auto_window: Optional[str] = None,
                        days: Optional[int] = None,
                        year: Optional[int] = None,
                        frequency: Optional[str] = None,
                        tz=pytz.utc,
                        time_type: str = 'datetime') \
        -> List[Tuple[kdatelike, kdatelike]]:

    assert time_type in ['date', 'timestamp']
    vargs = vars(parsed_args)

    w_start = ku.coalesce(window_start, vargs.get('window_start'))
    w_end = ku.coalesce(window_end, vargs.get('window_end'))
    w_date = ku.coalesce(window_date, vargs.get('window_date'))
    frequency = ku.coalesce(frequency, vargs.get('frequency'), 'none')
    auto_window = ku.coalesce(auto_window, vargs.get('auto_window'))
    days = ku.coalesce(days, vargs.get('days'))
    year = ku.coalesce(year, vargs.get('year'))
    tz = ku.coalesce(tz, vargs.get('timezone'), pytz.utc)
    weekdays = vargs['weekdays']
    last_weekday = vargs['last_weekday']

    # test that no incompatible arguments are used
    test, message = ktw.validate_time_window(window_start=w_start,
                                             window_end=w_end,
                                             window_date=w_date,
                                             auto_window=auto_window,
                                             days=days,
                                             year=year,
                                             weekdays=weekdays,
                                             last_weekday=last_weekday,
                                             frequency=frequency)
    if not test:
        raise argparse.ArgumentTypeError(f"'days' must be used with 'window_start' or "
                                         f"'window_end'")

    window_slices = ktw.decode_time_window_slices(window_start=w_start,
                                                  window_end=w_end,
                                                  window_date=w_date,
                                                  auto_window=auto_window,
                                                  days=days,
                                                  year=year,
                                                  weekdays=weekdays,
                                                  last_weekday=last_weekday,
                                                  frequency=frequency,
                                                  tz=tz,
                                                  time_type=time_type)
    return window_slices
