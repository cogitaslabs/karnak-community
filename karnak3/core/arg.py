from typing import Optional, Dict
import argparse
import datetime
import dateutil.parser
import pytz
import os


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
