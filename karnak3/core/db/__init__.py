import contextlib
import numbers
from abc import abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Union, Optional, Tuple, Any
import collections.abc
import re
import datetime

import sqlalchemy.pool
import sqlparams
import pandas as pd
import pyarrow as pa

import karnak3.core.log as kl
import karnak3.core.util as ku

_logger = kl.KLog(__name__)


def convert_paramstyle_numeric_to_plain(query: str, params: list) -> str:
    """Beware of sql injection: use results only for logging.
    Also, not very precise for non-string types.
    """

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
        return convert_paramstyle_numeric_to_plain(_sql, _params), None
    else:
        converter = sqlparams.SQLParams(_in_style, _out_style)
        return converter.format(sql, params)


class KWrappedFuture:
    def __init__(self, future: Future, metadata: Any = None):
        self.future = future
        self.metadata = metadata

    def result(self, timeout=None):
        return self.future.result(timeout=timeout)

    def exception(self, timeout=None):
        return self.future.exception(timeout=timeout)

    def cancel(self,):
        return self.future.cancel()

    def running(self) -> bool:
        return self.future.running()

    def done(self) -> bool:
        return self.future.done()

    def cancelled(self) -> bool:
        return self.future.cancelled()


class KPandasDataFrameFuture(KWrappedFuture):
    def __init__(self, future: Future, engine: 'KSqlAlchemyEngine', metadata: str = None):
        super().__init__(future=future)
        self.engine = engine

    @abstractmethod
    def to_df(self, timeout=None, save_memory: bool = False) -> Optional[pd.DataFrame]:
        pass


class KPandasDataFrameConstantFuture(KPandasDataFrameFuture):
    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df
        with ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: df)
            super().__init__(future, None, None)

    def to_df(self, timeout=None, save_memory: bool = False) -> Optional[pd.DataFrame]:
        return self.df


class KArrowTableFuture(KPandasDataFrameFuture):
    @abstractmethod
    def to_table(self, timeout=None) -> Optional[pd.DataFrame]:
        pass


class KarnakDBException(ku.KarnakException):
    pass



class KSqlAlchemyEngine:
    def __init__(self, engine_name: str,
                 paramstyle_client: str,
                 paramstyle_driver: str,
                 async_enabled: bool = False):
        self.engine_name = engine_name
        self.paramstyle_client = paramstyle_client
        self.paramstyle_driver = paramstyle_driver
        self.connection_pool: Optional[sqlalchemy.pool.Pool] = None
        self.set_paramstyle_client(paramstyle_client)
        self.async_enabled: bool = async_enabled
        assert paramstyle_driver in ['qmark', 'numeric', 'named', 'format', 'pyformat']

    def set_paramstyle_client(self, paramstyle: str):
        assert paramstyle in ['qmark', 'numeric', 'named', 'format', 'pyformat']
        self.paramstyle_client = paramstyle

    def set_connection_pool(self, pool: sqlalchemy.pool.Pool):
        self.connection_pool = pool

    @abstractmethod
    def _new_connection(self): pass

    def _connection(self):
        if self.connection_pool is not None:
            conn = self.connection_pool.connect()
        else:
            conn = self._new_connection()
        return conn

    @abstractmethod
    def _result_pd(self, cursor, result) -> Optional[pd.DataFrame]:
        pass

    def _result_pa(self, cursor, result) -> Optional[pa.Table]:
        raise ku.KarnakInternalError('method not implememented: _result_pa')

    def _result_pd_async(self, cursor, result) -> KPandasDataFrameFuture:
        raise ku.KarnakInternalError('method not implememented: _result_pd_async')

    def _result_pa_async(self, cursor, result) -> KArrowTableFuture:
        raise ku.KarnakInternalError('method not implememented: _result_pa_async')

    # TEST compatibility with other libs
    def test_libs_compatibility(self):
        from sklearn.preprocessing import MinMaxScaler, PowerTransformer
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        scaler = MinMaxScaler()
        xpto = scaler.fit(data)
        kl.trace('****** test_libs compatibility ok ****** ')

    def _engine_short_description(self) -> str:
        return self.engine_name

    def _convert_sql(self, sql: str,
                     params: Union[dict, list, None] = None,
                     paramstyle: str = None) -> Tuple[str, Union[dict, list, None]]:
        _paramstyle = paramstyle if paramstyle is not None else self.paramstyle_client

        sql_one_line = ' '.join(sql.split())
        _engine_str = self._engine_short_description()
        _logger.debug(f'running query on {_engine_str}, sql: {sql_one_line}, '
                      f'params {params}')
        plain_sql, _ = convert_paramstyle(sql_one_line, params, in_style=_paramstyle,
                                          out_style='plain')
        _logger.debug(f'plain query: {plain_sql}')
        _sql, _params = convert_paramstyle(sql_one_line, params, in_style=_paramstyle,
                                           out_style=self.paramstyle_driver)
        return _sql, _params

    def _cursor_execute(self, cursor,
                        sql: str,
                        params: Union[dict, list, None] = None):
        return cursor.execute(sql, params)

    def _cursor_params(self) -> dict:
        return {}

    def select_pa(self, sql: str,
                  params: Union[dict, list, None] = None,
                  paramstyle: str = None) -> pa.Table:
        _sql, _params = self._convert_sql(sql=sql, params=params, paramstyle=paramstyle)
        with contextlib.closing(self._connection()) as conn:
            with conn.cursor() as cursor:
                result = self._cursor_execute(cursor, _sql, _params)
                result_pa = self._result_pa(cursor, result)
                del result
                return result_pa

    def select_pd(self, sql: str,
                  params: Union[dict, list, None] = None,
                  paramstyle: str = None) -> pd.DataFrame:
        _sql, _params = self._convert_sql(sql=sql, params=params, paramstyle=paramstyle)
        with contextlib.closing(self._connection()) as conn:
            with conn.cursor() as cursor:
                result = self._cursor_execute(cursor, _sql, _params)
                # self.test_libs_compatibility()
                result_pd = self._result_pd(cursor, result)
                del result
                # self.test_libs()
                return result_pd

    def select_pa_async(self, sql: str,
                        params: Union[dict, list, None] = None,
                        paramstyle: str = None) -> KArrowTableFuture:
        _sql, _params = self._convert_sql(sql=sql, params=params, paramstyle=paramstyle)
        with contextlib.closing(self._connection()) as conn:
            with conn.cursor() as cursor:
                result = self._cursor_execute(cursor, _sql, _params)
                wrapped_future = self._result_pa_async(cursor, result)
                del result
                return wrapped_future

    def select_pd_async(self, sql: str,
                        params: Union[dict, list, None] = None,
                        paramstyle: str = None) -> KPandasDataFrameFuture:
        _sql, _params = self._convert_sql(sql=sql, params=params, paramstyle=paramstyle)
        _cursor_params = self._cursor_params()
        with contextlib.closing(self._connection()) as conn:
            with conn.cursor(**_cursor_params) as cursor:
                result = self._cursor_execute(cursor, _sql, _params)
                wrapped_future = self._result_pd_async(cursor, result)
                del result
                return wrapped_future
