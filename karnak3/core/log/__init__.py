import sys
import logging
from typing import Union, Optional, Dict
import karnak3.core.profiling as kprof

#
# Logging
#

# global karnak logger
karnak_logger = logging.getLogger('karnak')

_log_level_map: Dict[str, int] = {}

_level_name_map = {
    'NOTS': 0,
    'TRAC': 0,
    'DEBU': 1,
    'INFO': 2,
    'WARN': 3,
    'ERRO': 4,
    'CRIT': 5
}

_process = None

_DEFAULT_LOG_FORMAT = '%(levelname)-5.5s|%(asctime)-15s|%(message)s'

# logging.basicConfig(format=_DEFAULT_LOG_FORMAT)


def set_format(fmt: Optional[str] = None):
    _format = fmt if fmt is not None else _DEFAULT_LOG_FORMAT
    logging.basicConfig(format=_format, force=True)


def init(level: Union[int, str, None] = None):
    set_format()
    if level is not None:
        set_level(level)


def _normalize_level(level: Union[int, str]) -> Optional[int]:
    _level = None
    if isinstance(level, int):
        _level = level
    elif isinstance(level, str):
        _level = _level_name_map.get(level.upper()[:4])

    if _level is None:
        karnak_logger.error(f'invalid log level: {level}')
    return _level


def _get_py_level(level: int) -> int:
    return level * 10 if level > 0 else 1


def _default_level(module: str = 'karnak') -> int:
    if _log_level_map.get(module) is not None:
        _level = _log_level_map.get(module)
    elif _log_level_map.get('karnak') is not None:
        _level = _log_level_map.get('karnak')
    else:
        _level = 'INFO'
    normalized_level: int = _normalize_level(_level)
    return normalized_level


def set_level(level: Union[int, str], module: str = 'karnak'):
    _level = _normalize_level(level)
    _py_level = _get_py_level(_level)
    if _level is not None:
        _logger = logger(module)
        _logger.setLevel(level=_py_level)
        global _log_level_map
        _log_level_map[module] = _level


def logger(module: str = 'karnak'):
    _logger = logging.getLogger(module)
    return _logger


def get_log_threshold(level: Union[int, str],
                      logger_: Optional[logging.Logger] = None) -> (int, bool):
    _logger = logger_ if logger_ is not None else karnak_logger
    _level = _normalize_level(level)
    _py_level = _get_py_level(_level)
    _should_log = _logger.isEnabledFor(_py_level)
    return _py_level, _should_log


def log(level: Union[int, str], message, *args, memory: bool = False, exc_info=None,
        logger_: Optional[logging.Logger] = None):
    _logger = logger_ if logger_ is not None else karnak_logger
    _py_level, _should_log = get_log_threshold(level, logger_)
    _message = message
    if memory and _should_log:
        _message += f' (mem {kprof.memory_usage_str()})'
    _logger.log(_py_level, _message, *args, exc_info=exc_info)


def trace(message, *args, memory: bool = False):
    log('TRACE', message, *args, memory=memory)


def debug(message, *args, memory: bool = False):
    log('DEBUG', message, *args, memory=memory)


def info(message, *args, memory: bool = False):
    log('INFO', message, *args, memory=memory)


def warning(message, *args, memory: bool = False):
    log('WARN', message, *args, memory=memory)


def warn(message, *args, memory: bool = False):
    warning(message, *args, memory=memory)


def error(message, *args, memory: bool = False):
    log('ERROR', message, *args, memory=memory)


def critical(message, *args, memory: bool = False):
    log('CRITICAL', message, *args, memory=memory)


def exception(message, *args, exc_info: Exception = None, memory: bool = False):
    _message = message
    if exc_info is None:
        _exc_info = sys.exc_info()
    else:
        _exc_info = exc_info
        _message += f' (exception: {str(_exc_info)})'
    log('ERROR', _message, *args, memory=memory, exc_info=_exc_info)


class KLog:
    def __init__(self, module='karnak', level: Union[int, str, None] = None):
        self.level=level
        self.module = module
        self.logger: Optional[logging.Logger] = None

    def lazy_init(self):
        if self.logger is None:
            self.level = _default_level(self.module) if self.level is None else _normalize_level(self.level)
            self.logger = logger(self.module)
            set_level(self.level, self.module)

    def log(self, level: Union[int, str], message, memory: bool = False, exc_info=None):
        self.lazy_init()
        log(level=level, message=message, memory=memory, exc_info=exc_info,
            logger_=self.logger)

    def trace(self, message, memory: bool = False):
        self.log('TRACE', message, memory=memory)

    def debug(self, message, memory: bool = False):
        self.log('DEBUG', message, memory=memory)

    def info(self, message, memory: bool = False):
        self.log('INFO', message, memory=memory)

    def warning(self, message, memory: bool = False):
        self.log('WARN', message, memory=memory)

    def warn(self, message, memory: bool = False):
        self.warn(message, memory=memory)

    def error(self, message, memory: bool = False):
        self.log('ERROR', message, memory=memory)

    def critical(self, message, memory: bool = False):
        self.log('CRITICAL', message, memory=memory)




