from .decorators import *
from .path import *
from .compression import *
from .string import *
from .collections import *
from .datetime import *
from .tempfile import *
from .numeric import *
from .inspect import *

import karnak3.core.log as kl


class KarnakException(Exception):
    pass


class KarnakInternalError(KarnakException):
    def __init__(self, message: Optional[str] = None):
        _message = 'Karnak Internal Error'
        if message is not None:
            _message += ': ' + message
        super().__init__(_message)


class KarnakNotImplementedException(KarnakException):
    def __init__(self, message: Optional[str] = None):
        caller_name = inspect_caller_name(levels=2)
        _partial_message = 'not implemented' if message is None else message
        _message = f'{caller_name}: {_partial_message}'
        super().__init__(_message)

#
# Termination
#

def terminate(message, exit_code=1):
    kl.warn("TERMINATING: " + message)
    exit(exit_code)


def terminate_assert_list(value, lst: list, varname):
    if value not in lst:
        terminate(f"{varname} is '{value}' but should be in {str(lst)}")
