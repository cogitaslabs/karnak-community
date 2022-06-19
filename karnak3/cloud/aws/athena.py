import pandas as pd
import pyathena
import pyathena.pandas.util
import pyarrow

from pyathena.pandas.cursor import PandasCursor
from pyathena.pandas.async_cursor import AsyncPandasCursor
try:
    # requires pyathena>=2.7.0
    from pyathena.arrow.cursor import ArrowCursor
    from pyathena.arrow.async_cursor import AsyncArrowCursor
except:
    pass

import pyathenajdbc
import pyathenajdbc.util
from typing import Optional, Dict, Any, Union

import karnak.util.log as klog
from karnak3.core.db import KSqlAlchemyEngine, KPandasDataFrameFuture, KArrowTableFuture, KarnakDBException
from karnak3.core.db import KPandasDataFrameConstantFuture  # convenience for lib users. do not remove
import karnak3.core.arg as karg
import karnak3.cloud.aws as kaws
import karnak3.core.util as ku
import karnak3.core.log as kl
from karnak3.core.config import coalesce_config

# _paramstyle = 'numeric'


class AthenaConfig:
    def __init__(self, region: str,
                 default_database: Optional[str] = None,
                 workgroup: Optional[str] = None,
                 output_location: Optional[str] = None):

        self.region = ku.coalesce(region, kaws.aws_default_region())
        self.default_database = coalesce_config(default_database, 'ATHENA_DEFAULT_DATABASE')
        self.workgroup = coalesce_config(workgroup, 'ATHENA_WORKGROUP')
        self.output_location = coalesce_config(output_location, 'ATHENA_OUTPUT_LOCATION')
        super().__init__()


class KAthenaPandasDataFrameFuture(KPandasDataFrameFuture):
    def to_df(self, timeout=None, save_memory: bool = False) -> Optional[pd.DataFrame]:
        result_set = self.future.result(timeout)
        df = result_set.as_pandas()
        # noinspection PyTypeChecker
        _engine: AthenaEngine = self.engine
        _engine.inspect_query_execution(result_set, df)
        return df


class KAthenaArrowTableFuture(KArrowTableFuture):
    def to_table(self, timeout=None) -> Optional[pd.DataFrame]:
        result_set = self.future.result(timeout)
        pat = result_set.as_arrow()
        # noinspection PyTypeChecker
        _engine: AthenaEngine = self.engine
        _engine.inspect_query_execution(result_set, pat)
        return pat

    def to_df(self, timeout=None, save_memory: bool = False) -> Optional[pd.DataFrame]:
        pat = self.to_table()
        if pat is not None:
            # df = pat.to_pandas()
            if save_memory:
                df = pat.to_pandas(self_destruct=True, split_blocks=True)
            else:
                df = pat.to_pandas()
            del pat
            return df
        else:
            return None


class AthenaEngine(KSqlAlchemyEngine):
    def __init__(self, config: AthenaConfig,
                 paramstyle: str = 'numeric',
                 mode: str = 'rest',
                 async_enabled: bool = False,
                 unload: bool = False,
                 async_max_workers: Optional[int] = None):
        # Modes:
        # - rest: fast rest api
        # - csv: rest with slower cursor (better formatting for some types)
        # - jdbc: jdbc interface
        assert mode in ['rest', 'csv', 'jdbc', 'pandas', 'arrow']

        super().__init__('athena', paramstyle_client=paramstyle,
                         paramstyle_driver='pyformat', async_enabled=async_enabled)
        self.config = config
        self.mode = mode
        self.unload = unload
        self.async_max_workers = async_max_workers

    def _new_connection(self):
        config = self.config
        if self.mode == 'jdbc':
            conn_params = {'Workgroup': config.workgroup,
                           'AwsRegion': config.region,
                           'S3OutputLocation': config.output_location}
            if config.default_database is not None:
                conn_params['Schema'] = config.default_database

            conn = pyathenajdbc.connect(**conn_params)

        else:  # rest, csv
            assert self.mode in ['rest', 'csv', 'pandas', 'arrow']

            conn_params = {'work_group': config.workgroup,
                           'region_name': config.region,
                           #'output_location': config.output_location,
                           's3_staging_dir': config.output_location,
                           }
            if config.default_database is not None:
                conn_params['schema_name'] = config.default_database

            if self.mode in ['csv', 'pandas'] and not self.async_enabled:
                conn_params['cursor_class'] = PandasCursor
            elif self.mode == 'pandas' and self.async_enabled:
                conn_params['cursor_class'] = AsyncPandasCursor
            elif self.mode == 'arrow' and not self.async_enabled:
                conn_params['cursor_class'] = ArrowCursor
            elif self.mode == 'arrow' and self.async_enabled:
                conn_params['cursor_class'] = AsyncArrowCursor

            if self.unload:
                # requires pyathena >= 1.9 (Pandas) or pyathena >=1.7 (arrow)
                conn_params['cursor_kwargs'] = {'unload': True}

            conn = pyathena.connect(**conn_params)

        return conn

    def inspect_query_execution(self, _result, data: Union[pd.DataFrame, pyarrow.Table]):
        state = _result.state
        if state == 'FAILED':
            raise KarnakDBException(f'athena query failed: {_result.state_change_reason}')

        if data is not None:
            result_msg = f'query returned {len(data)} rows'
        else:
            result_msg = f'query returned empty data structure'

        if self.mode != 'jdbc':
            mode_detail_str = f'{self.mode}'
            if self.async_enabled:
                mode_detail_str += ' async'
            if self.unload:
                mode_detail_str += ' unload'

            result_msg += f', data scanned ({mode_detail_str}): ' \
                          f'{_result.data_scanned_in_bytes / (1024 * 1024.0):.2f}' \
                          f' MB, total query time ' \
                          f'{_result.total_execution_time_in_millis / 1000.0:.3f}s'
        klog.debug(result_msg)

    def _engine_short_description(self) -> str:
        _async_str = ' async' if self.async_enabled else ''
        _unload_str = ' unload' if self.unload else ''
        return f'{self.engine_name} ({self.mode}{_async_str}{_unload_str})'

    def _result_pa(self, cursor, result) -> Optional[pyarrow.Table]:
        if self.mode in ['arrow'] and not self.async_enabled:
            pat = result.as_arrow()
        elif self.mode in ['arrow'] and self.async_enabled:
            result_tuple = result
            query_id, future = result_tuple
            result = future.result()
            pat = result.as_arrow()
        else:
            raise ku.KarnakInternalError()
        self.inspect_query_execution(result, pat)
        return pat

    def _result_pd(self, cursor, result) -> Optional[pd.DataFrame]:
        if self.mode in ['csv', 'pandas'] and not self.async_enabled:
            df = result.as_pandas()
        elif self.mode in ['arrow'] and not self.async_enabled:
            df = result.as_arrow().to_pandas()
        elif self.mode == 'rest':
            df = pyathena.pandas.util.as_pandas(result)
        elif self.mode in ['pandas', 'arrow'] and self.async_enabled:
            result_tuple = result
            query_id, future = result_tuple
            result = future.result()
            if self.mode == 'pandas':
                df = result.as_pandas()
            else:
                df = result.as_arrow().to_pandas()
        elif self.mode == 'jdbc':
            df = pyathenajdbc.util.as_pandas(result)
        else:
            raise ku.KarnakInternalError()

        self.inspect_query_execution(result, df)
        return df

    def _result_pd_async(self, cursor, result) -> KPandasDataFrameFuture:
        if not self.async_enabled:
            raise ku.KarnakInternalError('async mode is not enabled')
        result_tuple = result
        query_id, future = result_tuple

        if self.mode == 'pandas':
            wrapped_future = KAthenaPandasDataFrameFuture(future, metadata=query_id, engine=self)
        elif self.mode == 'arrow':
            wrapped_future = KAthenaArrowTableFuture(future, metadata=query_id, engine=self)
        else:
            raise ku.KarnakInternalError('invalid mode')
        return wrapped_future

    def _result_pa_async(self, cursor, result) -> KArrowTableFuture:
        assert self.mode == 'arrow'
        # return correct type when mode == arrow
        # noinspection PyTypeChecker
        wrapped_future: KArrowTableFuture = self._result_pd_async(cursor, result)
        return wrapped_future

    def _cursor_execute(self, cursor,
                        sql: str,
                        params: Union[dict, list, None] = None):
        # workaround for null in float and double types
        if self.async_enabled and not self.unload and self.mode == 'pandas':
            # https://github.com/laughingman7743/PyAthena/issues/204
            return cursor.execute(sql, params, na_values=[''])
        else:
            return super()._cursor_execute(cursor, sql, params)

    def _cursor_params(self) -> dict:
        _params = {}
        if self.async_max_workers:
            _params['max_workers'] = self.async_max_workers
        return _params


def get_runtime_config(args: Optional[Dict[str, str]] = None):
    """read configuration from arguments or environment"""

    default_database = karg.best_arg('athena_default_database', args)
    region = karg.best_arg('athena_region', args, default=kaws.aws_default_region())
    workgroup = karg.best_arg('athena_workgroup', args)
    output_location = karg.best_arg('athena_output_location', args)

    config = AthenaConfig(region=region,
                          default_database=default_database,
                          workgroup=workgroup,
                          output_location=output_location)
    return config
