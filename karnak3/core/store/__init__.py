from abc import abstractmethod
from typing import Optional, Dict, Union, Type

import pandas as pd
from urllib.request import pathname2url
from urllib.parse import urljoin

import karnak3.core.util as ku
import karnak3.core.engine_registry as kcer


class KStoreRegistry(kcer.KarnakUrlRegistry):
    def __init__(self):
        super().__init__(name='store')

    def register(self, engine: 'KStoreEngine'):
        super().register(engine)

    def get_engine_by_key(self, key: str) -> Optional['KStoreEngine']:
        return super().get_engine_by_key(key=key)

    def get_engine(self, resource_name: str,
                   install_missing: bool = True,
                   allow_not_found: bool = True) -> Optional['KStoreEngine']:
        return super().get_engine(resource_name=resource_name,
                                  install_missing=install_missing,
                                  allow_not_found=allow_not_found)

    def get_missing_engine(self, resource: str) -> Optional['KStoreEngine']:
        key = self.get_key_from_url(resource)
        engine = None
        if key == 's3':
            import karnak3.cloud.aws.s3
            return self.get_engine(resource, install_missing=False, allow_not_found=True)
        # elif p == 'gs':
        #     import karnak.gcp.store.gs
        #     return cls.engine(uri, False)
        # elif p == 'file':
        #     import karnak.store.file
        #    return cls.engine(uri, False)
        else:
            return None


_store_registry = KStoreRegistry()


def get_engine(url: str, install_missing: bool = True) -> 'KStoreEngine':
    engine = _store_registry.get_engine(url, install_missing=install_missing,
                                        allow_not_found=False)
    if engine is None:
        raise ku.KarnakException(f'no store engine found for resource: {url}')
    return engine


class KStoreObject(kcer.KarnakUrlResource):

    @classmethod
    def is_localfile(cls) -> bool:
        return False

    @classmethod
    def temp_filepath(cls, suffix: Optional[str] = None) -> str:
        return ku.unique_temp_filepath(suffix=suffix)

    @abstractmethod
    def save_file(self, filepath: str) -> 'KStoreObject':
        pass

    @abstractmethod
    def load_file(self, filepath: Optional[str]) -> 'KStoreObject':
        pass


class KStoreConfig:
    pass


class KStoreEngine(kcer.KarnakUrlEngine):
    def __init__(self, scheme: str):
        super().__init__(scheme)
        _store_registry.register(self)

    @abstractmethod
    def build_store_object(self, url: str,
                           config: Optional[KStoreConfig] = None) -> KStoreObject:
        pass

    def save_file(self, url: str, filepath: str,
                  config: Optional[KStoreConfig] = None) -> KStoreObject:
        return self.build_store_object(url, config=config).save_file(filepath)

    def load_file(self, url: str, filepath: Optional[str],
                  config: Optional[KStoreConfig] = None) -> KStoreObject:
        return self.build_store_object(url, config=config).load_file(filepath)

    # @abstractmethod
    # def save_content(self,
    #                  url: str,
    #                  content: Union[bytes, str],
    #                  mime_type: str,
    #                  metadata: Optional[Dict[str, str]] = None,
    #                  config: Optional[KStoreConfig] = None) -> bool:
    #     pass
    #
    # @abstractmethod
    # def load_content(self,
    #                  url: str,
    #                  store_config: Optional[KStoreConfig] = None) -> bytes:
    #     pass
    #
    # @abstractmethod
    # def save_df(self,
    #             url: str,
    #             df: pd.DataFrame,
    #             store_config: Optional[KStoreConfig] = None) -> bool:
    #     pass



#
# def save_file(url: str, file_path: str,
#                 config: Optional[KStoreConfig] = None) -> bool:
#
#
#
# def download_file(url: str,
#                   file_path: Optional[str],
#                   config: Optional[KStoreConfig] = None) -> str:
#     pass
#
#
# def upload_content(url: str,
#                    content,
#                    mime_type: str,
#                    metadata: Optional[Dict[str, str]] = None,
#                    config: Optional[KStoreConfig] = None) -> bool:
#     pass
#
#
# def download_content(url: str,
#                      content,
#                      config: Optional[KStoreConfig] = None) -> bytearray:
#     pass
#
#
# def upload_df(url: str,
#               df: pd.DataFrame,
#               config: Optional[KStoreConfig] = None) -> bool:
#     pass
