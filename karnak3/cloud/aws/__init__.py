import os

import karnak3.core.util as ku


def aws_default_region():
    default_region = ku.coalesce(os.environ.get('KARNAK_AWS_REGION'),
                                 os.environ.get('AWS_REGION'),
                                 os.environ.get('AWS_DEFAULT_REGION'))
    return default_region
