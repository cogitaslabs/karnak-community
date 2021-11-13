import collections.abc
import numbers
from contextlib import contextmanager
from typing import Optional, Union

from redshift_connector import Connection

import karnak.util.log as klog
import pandas as pd
import redshift_connector
import sqlparams
import re
import datetime
import sqlalchemy.pool


class RedshiftConfig:
    def __init__(self, host: str, database: str, user: str, password: str):
        self.host = host
        self.database = database
        self.user = user
        self.password = password

    def connect(self):
        return redshift_connector.connect(host=self.host, database=self.database,
                                          user=self.user, password=self.password)


default_config: Optional[RedshiftConfig] = None
paramstyle = 'numeric'
redshift_connector.paramstyle = 'numeric'
connection_pool: Optional[sqlalchemy.pool.Pool] = None


def set_default_config(config: RedshiftConfig, create_connection_pool: bool = True):
    global default_config
    default_config = config
    if create_connection_pool:
        pool = sqlalchemy.pool.QueuePool(config.connect, max_overflow=10, pool_size=5)
        set_connection_pool(pool)


def set_parameter_style(style: str):
    assert style in ['qmark', 'numeric', 'named', 'format', 'pyformat']
    global paramstyle
    paramstyle = style


@contextmanager
def get_connection(config: Optional[RedshiftConfig] = None):
    global connection_pool, default_config

    if config is not None:
        conn = config.connect()
    elif connection_pool is not None:
        conn = connection_pool.connect()
    elif default_config is not None:
        conn = default_config.connect()
    else:
        return None

    try:
        yield conn
    finally:
        conn.close()


def set_connection_pool(pool: sqlalchemy.pool.Pool):
    global connection_pool
    connection_pool = pool


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
        regexp = ':' + si # + r'\D'
        value = to_str(params[i])
        new_query = re.sub(regexp,  str(value) + ' ', new_query)
    return new_query


def convert_paramstyle(sql, params: Union[dict, list, None],
                       in_style: str = None,
                       out_style: str = None) -> (str, Union[dict, list, None]):
    global paramstyle
    _out_style = out_style if out_style is not None else 'numeric'
    _in_style = in_style if in_style is not None else paramstyle
    if params is None or _in_style == _out_style:
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


def select_pd(sql: str, params: Union[dict, list, None] = None, config: Optional[RedshiftConfig] = None)\
        -> pd.DataFrame:
    sql_oneline = ' '.join(sql.split())
    klog.trace('running query on redshift, method: {}, params {}', sql_oneline, params)
    plain_sql, _ = convert_paramstyle(sql_oneline, params, out_style = 'plain')
    klog.trace(f'plain query: {plain_sql}')
    _sql, _params = convert_paramstyle(sql_oneline, params)

    with get_connection(config) as conn:
        with conn.cursor() as cursor:
            cursor.execute(_sql, args=_params)
            result = cursor.fetch_dataframe()
            return result
