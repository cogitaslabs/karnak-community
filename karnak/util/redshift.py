from contextlib import contextmanager
from typing import Optional, Union

import karnak.util.log as klog
import karnak.util.db as kdb

import pandas as pd
import redshift_connector
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
def _connect(config: Optional[RedshiftConfig] = None):
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


def select_pd(sql: str, params: Union[dict, list, None] = None, config: Optional[RedshiftConfig] = None)\
        -> pd.DataFrame:
    sql_one_line = ' '.join(sql.split())
    klog.trace('running query on redshift, method: {}, params {}', sql_one_line, params)
    plain_sql, _ = kdb.convert_paramstyle(sql_one_line, params, in_style=paramstyle, out_style='plain')
    klog.trace(f'plain query: {plain_sql}')
    _sql, _params = kdb.convert_paramstyle(sql_one_line, params, in_style=paramstyle, out_style='numeric')

    with _connect(config) as conn:
        with conn.cursor() as cursor:
            cursor.execute(_sql, args=_params)
            result = cursor.fetch_dataframe()
            return result
