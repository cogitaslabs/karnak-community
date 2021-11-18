import numbers
from typing import Optional, Union
import collections.abc

import sqlparams
import re
import datetime


def paramstyle_numeric_to_plain(query: str, params: list) -> str:
    """Beware of sql injection: use results only for logging. Also not very precise for non-string types"""

    def to_str(obj) -> str:
        if isinstance(obj, datetime.date):
            return f"date '{str(obj)}'"
        if isinstance(obj, datetime.datetime):
            return f"datetime '{str(obj)}'"
        elif isinstance(obj, str):
            if obj.startswith("'"):
                return f"{obj}"
            else:
                return f"'{obj}'"
        elif isinstance(obj, numbers.Number):
            return str(obj)
        elif isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str):
            seq_str = [f"'{e}'" for e in obj]
            return ','.join(seq_str)
        else:
            return f"'{str(obj)}'"

    new_query = query
    for i in range(len(params)):
        si = str(i + 1)
        regexp = ':' + si  # + r'\D'
        value = to_str(params[i])
        new_query = re.sub(regexp,  str(value) + ' ', new_query)
    return new_query


def convert_paramstyle(sql, params: Union[dict, list, None],
                       in_style: str,
                       out_style: str) -> (str, Union[dict, list, None]):
    assert in_style in ['qmark', 'numeric', 'named', 'format', 'pyformat']
    assert out_style in ['qmark', 'numeric', 'named', 'format', 'pyformat', 'plain']

    # global paramstyle
    _out_style = out_style
    _in_style = in_style
    if params is None or len(params) == 0 or _in_style == _out_style:
        return sql, params
    if _out_style == 'plain':
        if _in_style != 'numeric':
            converter = sqlparams.SQLParams(_in_style, 'numeric')
            _sql, _params = converter.format(sql, params)
        else:
            _sql, _params = sql, params
        return paramstyle_numeric_to_plain(_sql, _params), None
    else:
        converter = sqlparams.SQLParams(_in_style, _out_style)
        return converter.format(sql, params)