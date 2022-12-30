import os
from typing import Iterable, Optional, List

if (os.environ.get('KARNAK_DATAFRAME') is not None
        and os.environ.get('KARNAK_DATAFRAME').lower() == 'modin'):
    import modin.pandas as pandas
else:
    import pandas

from karnak3.core.util.numeric import precise_float_to_decimal

def is_empty(df: pandas.DataFrame) -> bool:
    return df is None or len(df) == 0


def not_empty(df: pandas.DataFrame) -> bool:
    return not is_empty(df)


def has_all_columns(df: pandas.DataFrame,
                    columns: Iterable[str]) -> bool:
    if df is None:
        return False
    has_columns = [c in df.columns.values for c in columns]
    return all(has_columns)


def ensure_columns(df: pandas.DataFrame,
                   columns: Iterable[str],
                   copy: bool = True) -> pandas.DataFrame:
    if copy:
        df = df.copy()
    # create non-existent columns
    for c in columns:
        if c not in df.columns.values:
            df[c] = None
    # reorder and drop extra columns
    df = df[columns]
    return df


def float_columns_to_decimal(df: pandas.DataFrame,
                             decimal_columns: List[str],
                             inplace: bool = False) -> pandas.DataFrame:
    if not inplace:
        df = df.copy()
    for c in decimal_columns:
        df[c] = df[c].map(precise_float_to_decimal)
    return df
