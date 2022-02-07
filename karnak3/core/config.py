import os
from typing import Optional


def get_config(variable_name: str) -> Optional[str]:

    # TODO: check in config file
    # TODO: support ssm prefix
    environ_var_name = 'KARNAK_' + variable_name.upper()
    return os.environ.get(environ_var_name)


def coalesce_config(value: Optional[str], variable_name: str) -> Optional[str]:
    if value is not None:
        return value
    return get_config(variable_name)
