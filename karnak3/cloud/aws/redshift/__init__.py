from typing import Optional, Dict
import pandas as pd

import redshift_connector
import sqlalchemy.pool

from karnak3.core.db import KSqlAlchemyEngine
import karnak3.core.arg as karg

redshift_connector.paramstyle = 'numeric'


class RedshiftConfig:
    def __init__(self, host: str, database: str, user: str, password: str):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        super().__init__()


class RedshiftEngine(KSqlAlchemyEngine):
    def __init__(self, config: RedshiftConfig,
                 paramstyle_client: str = 'numeric',
                 create_pool: bool = False):
        super().__init__('redshift',
                         paramstyle_client=paramstyle_client,
                         paramstyle_driver='numeric')
        self.config = config
        if create_pool:
            self.connection_pool = sqlalchemy.pool.QueuePool(self._new_connection,
                                                             max_overflow=10, pool_size=5)

    def _new_connection(self):
        return redshift_connector.connect(host=self.config.host,
                                          database=self.config.database,
                                          user=self.config.user,
                                          password=self.config.password)

    def _result_pd(self, cursor, result) -> Optional[pd.DataFrame]:
        return cursor.fetch_dataframe()


def get_runtime_config(args: Optional[Dict[str, str]] = None):
    """read configuration from arguments or environment"""

    host = karg.best_arg('redshift_host', args)
    database = karg.best_arg('redshift_database', args)
    user = karg.best_arg('redshift_user', args)
    password = karg.best_arg('redshift_password', args)

    config = RedshiftConfig(host=host, database=database, user=user, password=password)
    return config
