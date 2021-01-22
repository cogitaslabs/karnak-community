import karnak.util.log as klog

import pandas as pd
import contextlib
import pyathena
import pyathena.pandas.util
from pyathena.pandas.cursor import PandasCursor
import pyathenajdbc
import pyathenajdbc.util
from typing import Optional


def _select_pd_jdbc(sql: str, aws_region: str, database: Optional[str] = None,
                    workgroup: Optional[str] = None, s3_output_location: Optional[str] = None) -> pd.DataFrame:
    klog.trace('running query on athena, method jdbc: {}', ' '.join(sql.split()))
    if klog.log_level > 0:
        klog.debug('running query on athena, method rest')

    params = {'Workgroup': workgroup,
              'AwsRegion': aws_region,
              'S3OutputLocation': s3_output_location}
    if database is not None:
        params['Schema'] = database

    with contextlib.closing(pyathenajdbc.connect(**params)) as conn:
        with contextlib.closing(conn.cursor()) as cursor:
            results = cursor.execute(sql)
            klog.trace('query executed')

            if klog.log_level > 0:
                klog.debug('query execution completed.')
            df = pyathenajdbc.util.as_pandas(results)
            klog.trace('query results read into dataframe with {} rows.', len(df))
            return df


def _select_pd_rest(sql: str, aws_region: str, database=None, workgroup=None, s3_output_location=None,
                    method='rest') -> pd.DataFrame:
    assert method in ['rest', 'csv']
    klog.trace('running query on athena, method {}: {}', method, ' '.join(sql.split()))
    if klog.log_level > 0:
        klog.debug('running query on athena, method {}', method)

    params = {'work_group': workgroup,
              'region_name': aws_region,
              'output_location': s3_output_location}
    if database is not None:
        params['schema_name'] = database
    if method == 'csv':
        params['cursor_class'] = PandasCursor

    with contextlib.closing(pyathena.connect(**params)) as conn:
        with contextlib.closing(conn.cursor()) as cursor:
            results = cursor.execute(sql)
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


def select_pd(sql: str, aws_region: str, database: Optional[str] = None,
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
        return _select_pd_rest(sql=sql, aws_region=aws_region, database=database,
                               workgroup=workgroup, s3_output_location=s3_output_location, method=method)
    else:  # 'jdbc'
        return _select_pd_jdbc(sql=sql, aws_region=aws_region, database=database,
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
