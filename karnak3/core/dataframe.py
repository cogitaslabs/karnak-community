import os

if (os.environ.get('KARNAK_DATAFRAME') is not None
        and os.environ.get('KARNAK_DATAFRAME').lower() == 'modin'):
    import modin.pandas as pandas
else:
    import pandas
