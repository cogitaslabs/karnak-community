import karnak.util.log as klog
import karnak.util.db as kdb

import pandas as pd
import contextlib
import pyathena
import pyathena.pandas.util
from pyathena.pandas.cursor import PandasCursor
import pyathenajdbc
import pyathenajdbc.util
from typing import Optional, Dict, Any, Union

paramstyle = 'numeric'


def set_parameter_style(style: str):
    assert style in ['qmark', 'numeric', 'named', 'format', 'pyformat']
    global paramstyle
    paramstyle = style


def _select_pd_jdbc(sql: str, aws_region: str, database: Optional[str] = None,
                    params: Union[dict, list, None] = None,
                    workgroup: Optional[str] = None, s3_output_location: Optional[str] = None) -> pd.DataFrame:
    sql_one_line = ' '.join(sql.split())
    klog.trace('running query on athena, method jdbc: {}', sql_one_line)
    plain_sql, _ = kdb.convert_paramstyle(sql_one_line, params, in_style=paramstyle, out_style='plain')
    klog.trace(f'plain query: {plain_sql}')
    if klog.log_level > 0:
        klog.debug('running query on athena, method jdbc')

    _sql, _params = kdb.convert_paramstyle(sql_one_line, params, in_style=paramstyle, out_style='pyformat')

    connection_params = {'Workgroup': workgroup,
                         'AwsRegion': aws_region,
                         'S3OutputLocation': s3_output_location}
    if database is not None:
        connection_params['Schema'] = database

    with contextlib.closing(pyathenajdbc.connect(**connection_params)) as conn:
        with contextlib.closing(conn.cursor()) as cursor:
            results = cursor.execute(_sql, _params)
            klog.trace('query executed')

            if klog.log_level > 0:
                klog.debug('query execution completed.')
            df = pyathenajdbc.util.as_pandas(results)
            klog.trace('query results read into dataframe with {} rows.', len(df))
            return df


def _select_pd_rest(sql: str, aws_region: str, database=None,
                    params: Union[dict, list, None] = None, workgroup=None, s3_output_location=None,
                    method='rest') -> pd.DataFrame:
    assert method in ['rest', 'csv']
    sql_one_line = ' '.join(sql.split())
    klog.trace('running query on athena, method {}: {}', method, sql_one_line)
    plain_sql, _ = kdb.convert_paramstyle(sql_one_line, params, in_style=paramstyle, out_style='plain')
    klog.trace(f'plain query: {plain_sql}')
    if klog.log_level > 0:
        klog.debug('running query on athena, method {}', method)

    _sql, _params = kdb.convert_paramstyle(sql_one_line, params, in_style=paramstyle, out_style='pyformat')

    connection_params = {'work_group': workgroup,
                         'region_name': aws_region,
                         'output_location': s3_output_location}
    if database is not None:
        connection_params['schema_name'] = database
    if method == 'csv':
        connection_params['cursor_class'] = PandasCursor

    with contextlib.closing(pyathena.connect(**connection_params)) as conn:
        with contextlib.closing(conn.cursor()) as cursor:
            results = cursor.execute(_sql, _params)
            klog.trace('query stats: data scanned: {:.2f} MB, total query time {:.3f}s'.format(
                results.data_scanned_in_bytes / (1024 * 1024.0),
                results.total_execution_time_in_millis / 1000.0))

            if klog.log_level > 0:
                klog.debug('query execution completed.')
            if method == 'csv':
                df = results.as_pandas()
            else:
                df = pyathena.pandas.util.as_pandas(results)
            klog.trace('query results converted to dataframe with {} rows.', len(df))
            return df


def select_pd(sql: str, aws_region: str, params: Union[dict, list, None] = None, database: Optional[str] = None,
              workgroup: Optional[str] = None, s3_output_location: Optional[str] = None,
              method: str = 'rest') -> pd.DataFrame:
    """Runs an sql query in AWS Athena and returns results as pandas dataframe.

        Params:
            sql: sql query which will be executed by Athena.
            database: athena database used as default (optional if database is explicit in the sql query)
            workgroup: athena database. Either workgroup ou output_path (or both) must be provided.
            s3_output_path: athena s3 output path.
            aws_region: aws s3 region name, e.g.: 'us_east-1'
            method: results download method. 'rest' or 'jdbc' are best suited for small results data;
                'csv' is suitable for larger results data.
    """

    assert method in ['jdbc', 'rest', 'csv']
    if method in ['rest', 'csv']:
        return _select_pd_rest(sql=sql, aws_region=aws_region, params=params, database=database,
                               workgroup=workgroup, s3_output_location=s3_output_location, method=method)
    else:  # 'jdbc'
        return _select_pd_jdbc(sql=sql, aws_region=aws_region, params=params, database=database,
                               workgroup=workgroup, s3_output_location=s3_output_location)


def test_fixture_pd(aws_region: str, workgroup: Optional[str] = None, s3_output_location: Optional[str] = None,
                    method: str = 'rest'):
    sql = """
        SELECT * FROM (VALUES
            (CAST ('str1' AS VARCHAR), 1, CAST(1 AS BIGINT), 1.0, TRUE, '["elem1", "elem2"]'),
            ('str2', 2, 2,  2.0, FALSE, '["elem3"]'),
            ('', 0, 0, 0.0, FALSE, '[]'),
            (NULL, NULL, NULL, NULL, NULL, NULL)
        )
            x(c_str, c_int, c_bigint, c_double, c_boolean, c_json_list);
"""
    return select_pd(sql, aws_region=aws_region, workgroup=workgroup, s3_output_location=s3_output_location,
                     method=method)
