import os
import pandas as pd
import pyathena
import pyathena.pandas.util
from pyathena.pandas.cursor import PandasCursor
import pyathenajdbc
import pyathenajdbc.util
from typing import Optional, Dict, Any, Union

import karnak.util.log as klog
from karnak3.core.db import KSqlAlchemyEngine
import karnak3.core.arg as karg
import karnak3.cloud.aws as kaws
import karnak3.core.util as ku
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


class AthenaEngine(KSqlAlchemyEngine):
    def __init__(self, config: AthenaConfig,
                 paramstyle: str = 'numeric',
                 mode: str = 'rest'):
        # Modes:
        # - rest: fast rest api
        # - csv: rest with slower cursor (better formatting for some types)
        # - jdbc: jdbc interface
        assert mode in ['rest', 'csv', 'jdbc']

        super().__init__('athena', paramstyle_client=paramstyle,
                         paramstyle_driver='pyformat')
        self.config = config
        self.mode = mode

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
            assert self.mode in ['rest', 'csv']

            conn_params = {'work_group': config.workgroup,
                           'region_name': config.region,
                           'output_location': config.output_location}
            if config.default_database is not None:
                conn_params['schema_name'] = config.default_database
            if self.mode == 'csv':
                conn_params['cursor_class'] = PandasCursor
            conn = pyathena.connect(**conn_params)

        return conn

    def _result_pd(self, cursor, result) -> Optional[pd.DataFrame]:
        if self.mode == 'csv':
            df = result.as_pandas()
        elif self.mode == 'rest':
            df = pyathena.pandas.util.as_pandas(result)
        elif self.mode == 'jdbc':
            df = pyathenajdbc.util.as_pandas(result)
        else:
            raise ku.KarnakInternalError()

        result_msg = f'query returned {len(df)} rows'
        if self.mode in ['rest', 'csv']:
            result_msg += f', data scanned: ' \
                          f'{result.data_scanned_in_bytes / (1024 * 1024.0):.2f}' \
                          f' MB, total query time ' \
                          f'{result.total_execution_time_in_millis / 1000.0:.3f}s'
        klog.debug(result_msg)

        return df


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
