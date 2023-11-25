import os
from typing import Optional

from karnak3.core.util import KarnakException


def get_config(variable_name: str) -> Optional[str]:

    # TODO: check in config file
    # TODO: support ssm prefix
    environ_var_name = 'KARNAK_' + variable_name.upper()
    return os.environ.get(environ_var_name)


def coalesce_config(value: Optional[str], variable_name: str) -> Optional[str]:
    if value is not None:
        return value
    return get_config(variable_name)


def coalesce_config_int(value: Optional[int], variable_name: str) -> Optional[int]:
    if value is not None:
        return value
    value_str = get_config(variable_name)
    try:
        value_int = int(value_str)
        return value_int
    except Exception as e:
        raise KarnakException(f'environment variable {variable_name} should be an integer.', e)


def coalesce_config_bool(value: Optional[bool], variable_name: str) -> Optional[bool]:
    if value is not None:
        return value
    value_str = get_config(variable_name)
    try:
        value_bool = bool(value_str)
        return value_bool
    except Exception as e:
        raise KarnakException(f'environment variable {variable_name} should be a boolean.', e)

