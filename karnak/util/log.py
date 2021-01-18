import sys
import logging
import os
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


def log_generic(level_str, level, message, *msg_fmt, file=None, force=False):
    if file is None:
        file = log_destination
    if force or level >= log_level:
        global log_simple
        str_msg_fmt = [str(o) for o in msg_fmt]
        msg = message.format(*str_msg_fmt)
        if log_simple:
            print(level_str + ':', msg, file=file)
        else:
            print(level_str, '|', datetime.datetime.now(), '|', msg, file=file)


def trace(message, *msg_fmt, file=None, force=False, mem=False):
    global process
    if mem and process is None:
        process = psutil.Process(os.getpid())
    if mem:
        log_generic('TRACE', 0, message, *msg_fmt, '|', process.memory_info().rss / 1024 ** 2, 'MB',
                   file=file, force=force)
    else:
        log_generic('TRACE', 0, message, *msg_fmt, '|', file=file, force=force)


def debug(message, *msg_fmt, file=None, force=False, mem=False):
    if mem:
        log_generic('DEBUG', 1, message, *msg_fmt, process.memory_info().rss / 1024 ** 2, 'MB',
                   file=file, force=force)
    else:
        log_generic('DEBUG', 1, message, *msg_fmt, file=file, force=force)


def info(message, *msg_fmt, file=None, force=False):
    log_generic('INFO ', 2, message, *msg_fmt, file=file, force=force)


def warn(message, *msg_fmt, file=None, force=False):
    log_generic('WARN ', 3, message, *msg_fmt, file=file, force=force)


def error(message, *msg_fmt, file=None, force=False):
    log_generic('ERROR', 4, message, *msg_fmt, file=file, force=force)


def exception(message, e: Exception, *msg_fmt, file=None, force=False):
    import traceback
    log_generic('ERROR', 4, message + ': ' + str(e), *msg_fmt, file=file, force=force)
    traceback.print_exc()

#
#
#