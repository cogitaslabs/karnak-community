from typing import Optional, Union

import pandas as pd
import sqlalchemy.pool

from . import *


_redshift_engine = KRedshiftEngine(None)  # noqa


class RedshiftConfig(RedshiftConfig):
    pass


def set_default_config(config: RedshiftConfig, create_connection_pool: bool = True):
    _redshift_engine.config = config
    if create_connection_pool:
        pool = sqlalchemy.pool.QueuePool(config.connect, max_overflow=10, pool_size=5)
        _redshift_engine.set_connection_pool(pool)


def set_parameter_style(style: str):
    _redshift_engine.set_paramstyle_client(style)


def set_connection_pool(pool: sqlalchemy.pool.Pool):
    _redshift_engine.set_connection_pool(pool)


def select_pd(sql: str,
              params: Union[dict, list, None] = None,
              config: Optional[RedshiftConfig] = None) -> pd.DataFrame:
    _engine = _redshift_engine
    if config is not None:
        _engine = KRedshiftEngine(config, paramstyle=_redshift_engine.paramstyle_client)
    return _engine.select_pd(sql, params)
