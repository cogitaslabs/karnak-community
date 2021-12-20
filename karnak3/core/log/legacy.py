import karnak3.core.log as nl


def set_log_level(level, buffered=False):
    nl.set_level(level)


def set_log_destination(file):
    nl.warn('set_log_destination is deprecated')


def loggeneric(level_str, level, message, *msg_fmt, file=None, force=False, mem=False):
    if file is not None:
        nl.warn('using FILE in log in deprecated')
    _level = 'INFO' if force else level
    _message = message.format(*msg_fmt)
    nl.log(level=_level, message=message, *msg_fmt)


def log_generic(level_str, level, message, *msg_fmt, file=None, force=False, mem=False):
    loggeneric(level_str, level, message, *msg_fmt, file=None, force=False, mem=False)


def trace(message, *msg_fmt, file=None, force=False, mem=True):
    loggeneric('TRACE', 0, message, *msg_fmt, mem=mem, force=force)


def logtrace(message, *msg_fmt, file=None, force=False, mem=True):
    loggeneric('TRACE', 0, message, *msg_fmt, mem=mem, force=force)


def debug(message, *msg_fmt, file=None, force=False, mem=False):
    loggeneric('DEBUG', 1, message, *msg_fmt, mem=mem, force=force)


def logdebug(message, *msg_fmt, file=None, force=False, mem=False):
    loggeneric('DEBUG', 1, message, *msg_fmt, mem=mem, force=force)


def info(message, *msg_fmt, file=None, force=False):
    loggeneric('INFO ', 2, message, *msg_fmt, file=file, force=force)


def loginfo(message, *msg_fmt, file=None, force=False):
    loggeneric('INFO ', 2, message, *msg_fmt, file=file, force=force)


def warn(message, *msg_fmt, file=None, force=False):
    loggeneric('WARN ', 3, message, *msg_fmt, file=file, force=force)


def logwarn(message, *msg_fmt, file=None, force=False):
    loggeneric('WARN ', 3, message, *msg_fmt, file=file, force=force)


def error(message, *msg_fmt, file=None, force=False):
    loggeneric('ERROR', 4, message, *msg_fmt, file=file, force=force)


def logerror(message, *msg_fmt, file=None, force=False):
    loggeneric('ERROR', 4, message, *msg_fmt, file=file, force=force)


def exception(message, e: Exception, *msg_fmt, file=None, force=False):
    import traceback
    loggeneric('ERROR', 4, message + ': ' + str(e), *msg_fmt, file=file, force=force)
    traceback.print_exc()


def logexception(message, e: Exception, *msg_fmt, file=None, force=False):
    exception(message, e, *msg_fmt, file=file, force=force)





