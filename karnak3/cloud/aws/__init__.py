import os
from typing import Dict, Any, Optional

import boto3
import threading

import karnak3.core.util as ku
from karnak3.cloud import CloudEngine


class AWSCloudEngine(CloudEngine):
    def __init__(self):
        super().__init__('aws')

    def start(self, force: bool = False):
        get_aws_client('s3')
        get_aws_client('sts')


_aws_cloud_engine = AWSCloudEngine()


#
# clients
#

_client_lock = threading.RLock()
_aws_client_cache: Dict[str, Any] = {}


def get_aws_client(service: str, aws_region: Optional[str] = None, use_cache: bool = False):
    if use_cache:
        client = _aws_client_cache.get(service)
        if client:
            return client

    _aws_region = ku.coalesce(aws_region, aws_default_region())
    with _client_lock:
        if _aws_region:
            client = boto3.client(service, region_name=_aws_region)
        else:
            client = boto3.client(service)
        if use_cache:
            _aws_client_cache[service] = client
        return client


def aws_default_region() -> Optional[str]:
    default_region = ku.coalesce(os.environ.get('KARNAK_AWS_REGION'),
                                 os.environ.get('AWS_REGION'),
                                 os.environ.get('AWS_DEFAULT_REGION'))
    return default_region

