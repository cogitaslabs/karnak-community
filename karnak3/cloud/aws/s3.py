from typing import Optional, Dict, Tuple

import pandas as pd
from botocore.client import BaseClient
from urllib.parse import urlparse

import karnak3.core.util as ku
import karnak3.cloud.aws as kcaws
import karnak3.core.store as kcs
from karnak3.core.store import KStoreEngine, KStoreConfig, KStoreObject


class KStoreS3ConfigS3(KStoreConfig):
    def __init__(self):
        self.aws_region: Optional[str] = None
        self.use_client_cache: bool = False
        self.allow_multipart: bool = True
        self.multipart_threshold: Optional[int] = None


class KStoreS3Object(KStoreObject):
    def __init__(self, url: str, engine: Optional['KStoreS3Engine'] = None,
                 config: Optional[KStoreS3ConfigS3] = None):
        super().__init__(resource_type='s3-object', url=url)
        self.bucket = self.netloc
        self.key = self.path
        self.engine = engine
        if engine is None:
            self.engine = kcs.get_engine(url)
        self.config = ku.coalesce(config, KStoreS3ConfigS3())

    def get_s3_client(self) -> BaseClient:
        return KStoreS3Engine.get_s3_client(config=self.config)

    def save_file(self, filepath: str) -> 'KStoreS3Object':
        client = self.get_s3_client()
        client.Bucket(self.bucket).save_file(filepath, self.key)
        return self

    def load_file(self, filepath: Optional[str]) -> 'KStoreS3Object':
        _filepath = filepath
        if _filepath is None:
            _filepath = self.temp_filepath()
        client = self.get_s3_client()
        client.Bucket(self.bucket).save_file(_filepath, self.key)
        return self


class KStoreS3Engine(KStoreEngine):
    def __init__(self):
        super().__init__('s3')

    @staticmethod
    def get_s3_client(config: Optional[KStoreS3ConfigS3] = None) -> BaseClient:
        _config = ku.coalesce(config, KStoreS3ConfigS3())
        client = kcaws.get_aws_client(service='s3', aws_region=_config.aws_region,
                                      use_cache=_config.use_client_cache)
        return client

    def build_store_object(self, url: str,
                           config: Optional[KStoreS3ConfigS3] = None) -> KStoreS3Object:
        return KStoreS3Object(url, engine=self, config=config)

    def save_file(self, url: str, filepath: str,
                  config: Optional[KStoreS3ConfigS3] = None) -> KStoreS3Object:
        return self.build_store_object(url, config=config).save_file(filepath)

    def load_file(self, url: str, filepath: Optional[str],
                  config: Optional[KStoreConfig] = None) -> KStoreS3Object:
        return self.build_store_object(url, config=config).load_file(filepath)


# register engine
_engine = KStoreS3Engine()
