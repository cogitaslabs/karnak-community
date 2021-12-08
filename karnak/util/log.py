import sys
import logging
import os
import threading

import psutil
import datetime

#
# Logging
#

LOG_FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=LOG_FORMAT)
log_level = 0
log_simple = False
log_destination = sys.stderr
process = None


def set_log_level(level, buffered=False):
    global log_level
    log_level = level
    logging.basicConfig(level=log_level * 10)
    if buffered:
        os.environ['PYTHONUNBUFFERED'] = ''
    else:
        os.environ['PYTHONUNBUFFERED'] = '1'


def set_log_destination(file):
    global log_destination
    log_destination = file


def get_logger(module):
    global log_level
    logger = logging.getLogger(module)
    logger.setLevel(log_level * 10)
    return logger


_log_lock = threading.Lock()


def log_generic(level_str, level, message, *msg_fmt, file=None, force=False, thread_safe=True):
    if file is None:
        file = log_destination
    if force or level >= log_level:
        global log_simple
        str_msg_fmt = [str(o) for o in msg_fmt]
        msg = message.format(*str_msg_fmt)

        def print_log():
            if log_simple:
                print(f'{level_str}:{msg}\n', file=file, end='')
            else:
                print(f'{level_str}|{datetime.datetime.now()}|{msg}\n', file=file, end='')

        if thread_safe:
            with _log_lock:
                print_log()
        else:
            print_log()


def trace(message, *msg_fmt, file=None, force=False, mem=False, thread_safe=True):
    global process
    if mem and process is None:
        process = psutil.Process(os.getpid())
    if mem:
        log_generic('TRACE', 0, message, *msg_fmt, '|', process.memory_info().rss / 1024 ** 2, 'MB',
                    file=file, force=force, thread_safe=thread_safe)
    else:
        log_generic('TRACE', 0, message, *msg_fmt, '|', file=file, force=force, thread_safe=thread_safe)


def debug(message, *msg_fmt, file=None, force=False, mem=False, thread_safe=True):
    if mem:
        log_generic('DEBUG', 1, message, *msg_fmt, process.memory_info().rss / 1024 ** 2, 'MB',
                    file=file, force=force)
    else:
        log_generic('DEBUG', 1, message, *msg_fmt, file=file, force=force, thread_safe=thread_safe)


def info(message, *msg_fmt, file=None, force=False, thread_safe=True):
    log_generic('INFO ', 2, message, *msg_fmt, file=file, force=force, thread_safe=thread_safe)


def warn(message, *msg_fmt, file=None, force=False, thread_safe=True):
    log_generic('WARN ', 3, message, *msg_fmt, file=file, force=force, thread_safe=thread_safe)


def error(message, *msg_fmt, file=None, force=False, thread_safe=True):
    log_generic('ERROR', 4, message, *msg_fmt, file=file, force=force, thread_safe=thread_safe)


def exception(message, e: Exception, *msg_fmt, file=None, force=False, thread_safe=True):
    import traceback
    log_generic('ERROR', 4, message + ': ' + str(e), *msg_fmt, file=file, force=force, thread_safe=thread_safe)
    traceback.print_exc()
