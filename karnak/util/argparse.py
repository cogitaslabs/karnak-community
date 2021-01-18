import argparse
import datetime
import dateutil.parser
import pytz
import os


def argparse_date(date_str):
    try:
        return datetime.datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise argparse.ArgumentTypeError(f'date {date_str} not in format YYYY-MM-DD')


def argparse_timestamp(s):
    try:
        return dateutil.parser.isoparse(s)
    except Exception:
        raise argparse.ArgumentTypeError(f'not a valid timestamp: {s}')


def argparse_timezone(s):
    try:
        return pytz.timezone(s)
    except Exception:
        raise argparse.ArgumentTypeError(f'not a valid timezone: {s}')


def argparse_dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")


def argparse_file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid file")
